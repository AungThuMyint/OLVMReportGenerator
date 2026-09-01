"""OLVM Report Generator - comprehensive inventory report tool (single file).

Connects to an OLVM/oVirt Engine v4 REST API, pulls inventory data
(datacenters, clusters, hosts, storage domains, networks, virtual machines
with their IPs / networks / storage) and produces a PDF report.

The engine REST API is reachable at:
    https://<engine>/ovirt-engine/api
and uses HTTP Basic authentication with the user login (e.g. admin@internal).

Usage:
    python OLVMReportGenerator.py [--user USER] [--output FILE.pdf] [--insecure]
        [--platform NAME] [--timeout SECONDS]
        --engine NAME --password PASS --name NAME --logo FILE
"""

import argparse
import getpass
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    PageBreak,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)


class OLVMError(Exception):
    """Raised when the engine API returns an error or cannot be reached."""


@dataclass
class OLVMClient:
    """Minimal client for the oVirt Engine v4 REST API."""

    engine: str
    username: str
    password: str
    insecure: bool = False
    timeout: int = 60
    _session: requests.Session = field(default_factory=requests.Session, init=False)

    def __post_init__(self) -> None:
        if "://" not in self.engine:
            self.engine = "https://" + self.engine.strip()
        self.engine = self.engine.rstrip("/")
        self._base_yield = f"{self.engine}/ovirt-engine/api"
        self._base = self._base_yield
        for h, v in (("Accept", "application/xml"), ("User-Agent", "OLVM-Report-Generator/1.0")):
            self._session.headers[h] = v
        if self.insecure:
            warnings.filterwarnings("ignore", message="Unverified HTTPS request")
            self._session.verify = False

    # ------------------------------------------------------------------ #
    # Low level helpers
    # ------------------------------------------------------------------ #
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None,
             force_list: bool = False) -> Dict[str, Any]:
        url = f"{self._base}{path}"
        try:
            resp = self._session.get(
                url,
                params=params,
                auth=HTTPBasicAuth(self.username, self.password),
                timeout=self.timeout,
            )
        except requests.exceptions.SSLError as exc:
            raise OLVMError(
                f"SSL certificate verification failed for {url}. "
                f"Use --insecure to skip verification. ({exc})"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OLVMError(f"Could not reach the engine at {url}: {exc}") from exc

        if resp.status_code == 401:
            raise OLVMError("Authentication failed. Check username and password.")
        if resp.status_code == 404:
            return {"data": [], "total": 0}
        if resp.status_code != 200:
            raise OLVMError(
                f"Engine returned HTTP {resp.status_code} for {url}: {resp.text[:500]}"
            )
        return self._parse_xml(resp.content, force_list=force_list)
    @staticmethod
    def _parse_xml(raw: bytes, force_list: bool = False) -> Dict[str, Any]:
        """Parse the oVirt REST XML body into a nested dict.

        If ``force_list`` is True, every child group of this element is
        stored as a list even when there is only a single child. This is
        used for collection root elements (e.g. ``<datacenters>``) so that
        a collection with a single member still yields a list entry.
        """
        try:
            from xml.etree import ElementTree as ET
        except Exception:
            ET = None
        if ET is None:
            raise OLVMError("ElementTree is not available - cannot parse engine response.")

        root = ET.fromstring(raw)
        tree: Dict[str, Any] = {"_tag": root.tag.split("}")[-1]}

        # Capture element attributes (e.g. id, href, name on oVirt elements).
        for aname, aval in (root.attrib or {}).items():
            tree[aname] = {"_text": str(aval)}

        children = [c for c in root if c.tag]
        if not children:
            tree["_text"] = (root.text or "").strip()

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for child in children:
            tag = child.tag.split("}")[-1]
            grouped.setdefault(tag, []).append(
                OLVMClient._parse_xml(ET.tostring(child)))

        for tag, items in grouped.items():
            if force_list or len(items) > 1:
                tree[tag] = items
            else:
                tree[tag] = items[0]
        return tree

    # ------------------------------------------------------------------ #
    # Resource listings
    # ------------------------------------------------------------------ #
    def get(self, path: str, params: Optional[Dict[str, Any]] = None,
            force_list: bool = False) -> Dict[str, Any]:
        """GET a single resource (returns dict) or collection (returns {'data': [...]})."""
        doc = self._get(path, params=params, force_list=force_list)
        list_key = None
        for key, value in doc.items():
            if isinstance(value, list):
                list_key = key
                break
        if list_key is not None:
            return {"data": doc[list_key], "total": doc.get("total", {"_text": len(doc[list_key])})}
        return doc

    def list(self, resource: str, max_results: int = 10000) -> List[Dict[str, Any]]:
        """List all entries of a collection resource (e.g. 'datacenters')."""
        doc = self._get_with_follow(f"/{resource}", True)
        entries = []
        data = doc.get("data")
        if data is None:
            return entries
        if isinstance(data, list):
            entries = data
        else:
            entries = [data]
        return self._normalize_list(entries)

    @staticmethod
    def _normalize_list(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            # Undo the single-item collapse so each entry keeps 'data' recursively
            e2 = dict(e)
            if "_tag" in e2 and e2.get("_tag") != "data" and "data" not in e2:
                pass
            out.append(e2)
        return out

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def text(self, d: Any) -> str:
        """Return the text content of an XML leaf (a dict with '_text')."""
        if isinstance(d, dict):
            return d.get("_text", "")
        return str(d or "")

    def lookup(self, d: Any, key: str, default: str = "") -> str:
        """Return nested text value using dotted path, e.g. 'vm.name'."""
        cur = d
        for part in key.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part, {})
            else:
                return default
        return self.text(cur)

    def id_of(self, ref: Any) -> str:
        return self.lookup(ref, "id")

    def follow(self, ref: Any, resource: str, **kw) -> Dict[str, Any]:
        """Fetch a referenced resource by href/id."""
        href = self.lookup(ref, "href")
        rid = self.lookup(ref, "id")
        if href:
            # convert absolute engine URL to relative path
            if self._base in href:
                path = href[len(self._base):]
            else:
                path = href
            try:
                return self.get(path)
            except OLVMError:
                pass
        if rid:
            try:
                return self.get(f"/{resource}/{rid}")
            except OLVMError:
                pass
        return {}

    # ------------------------------------------------------------------ #
    # High level report data
    # ------------------------------------------------------------------ #
    def _get_with_follow(self, path: str, force_list: bool = False) -> Dict[str, Any]:
        """Try to GET a sub-resource with follow=* and fall back to no follow.

        Some engines fail (HTTP 500) when processing follow=*, so the request
        is retried without it. Using follow enriches nested link data such as
        network names inside NICs and storage domains inside disks.
        """
        try:
            return self.get(path, params={"follow": "*"}, force_list=force_list)
        except OLVMError:
            return self.get(path, force_list=force_list)

    def gather(self) -> Dict[str, Any]:
        """Collect everything needed for the report."""
        report: Dict[str, Any] = {}

        # Engine version (from the API root, best effort)
        report["engine_version"] = ""
        try:
            root = self._get("")  # GET /api root element
            ver = root.get("product_info", {}) if isinstance(root, dict) else {}
            major = self.lookup(ver, "version.major")
            minor = self.lookup(ver, "version.minor")
            build = self.lookup(ver, "version.build")
            if minor:
                report["engine_version"] = "{}.{}".format(major, minor)
                if build:
                    report["engine_version"] += ".{}".format(build)
        except OLVMError:
            pass

        report["datacenters"] = self.list("datacenters")
        report["clusters"] = self.list("clusters")
        report["hosts"] = self.list("hosts")
        report["storagedomains"] = self.list("storagedomains")
        report["networks"] = self.list("networks")
        report["vms"] = self.list("vms")

        report["disks"] = self.list("disks")
        report["vnicprofiles"] = self.list("vnicprofiles")

        report["diskattachments"] = []
        for vm in report["vms"]:
            vid = self.lookup(vm, "id")
            href = self.lookup(vm, "href")
            if not vid and not href:
                continue
            base = f"/vms/{vid}" if vid else href

            # Virtual network interfaces
            try:
                vms_nics = self._get_with_follow(f"{base}/nics", True).get("data", [])
            except OLVMError:
                vms_nics = []
            vm["nics"] = {"nic": vms_nics}

            # Guest IPs (reported devices)
            try:
                vm_devs = self._get_with_follow(f"{base}/reporteddevices", True).get("data", [])
            except OLVMError:
                vm_devs = []
            vm["reported_devices"] = {"reported_device": vm_devs}

            # Disk attachments -> disk name/size/storage domain
            try:
                atts = self._get_with_follow(f"{base}/diskattachments", True).get("data", [])
            except OLVMError:
                atts = []
            for att in atts:
                report["diskattachments"].append(att)

        # Physical host network cards (link speed, status, bound network)
        report["host_nics"] = []
        for h in report["hosts"]:
            hid = self.lookup(h, "id")
            href = self.lookup(h, "href")
            if not hid and not href:
                continue
            hbase = f"/hosts/{hid}" if hid else href
            try:
                hnics = self._get_with_follow(f"{hbase}/nics", True).get("data", [])
            except OLVMError:
                hnics = []
            report["host_nics"].append({"host": {"id": {"_text": hid}} if hid else {},
                                        "nics": {"nic": hnics}})
        return report


HEADER_BG = colors.HexColor("#1f4e78")
SUBHEADER_BG = colors.HexColor("#2f6ea8")
ALT_ROW = colors.HexColor("#eef3f8")
ACCENT = colors.HexColor("#3b82c4")


# ------------------------------------------------------------------ #
# Helpers to work with the oVirt XML-to-dict shape produced by the client
# ------------------------------------------------------------------ #
def _txt(value: Any) -> str:
    """Return text content of a leaf (dict with '_text') or str(value)."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("_text", ""))
    return str(value)


def _leaf(data: Any, *keys: str) -> str:
    """Walk a nested dict using dotted keys and return the leaf text."""
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
        if cur is None:
            return ""
    return _txt(cur)


def _status(value: str) -> str:
    """Display a status value with the first letter capitalised (up -> Up)."""
    v = (value or "").strip()
    if not v:
        return "-"
    return v[0].upper() + v[1:]


def _lists(doc: Any, key: str) -> List[Dict[str, Any]]:
    """Return a list of dict entries under a key (handles 0/1/many children)."""
    entries = doc.get(key) if isinstance(doc, dict) else None
    if entries is None:
        return []
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    if isinstance(entries, dict):
        return [entries]
    return []


def _name(ref: Any) -> str:
    return _leaf(ref, "name")


def _fmt_size(size_str: str) -> str:
    try:
        b = float(size_str)
    except (TypeError, ValueError):
        return str(size_str)
    if b < 0:
        return str(size_str)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024
    return str(size_str)


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _size_or_dash(value: float) -> str:
    """Human-readable size, or '-' when 0/unknown."""
    if not value:
        return "-"
    return _fmt_size("{:.0f}".format(value))


def _avail_display(used: float, total: float, unit: str = "") -> str:
    """Human-readable 'available' for a resource based on used vs total.

    If the allocation exceeds the capacity (negative remainder), report a
    clear 'over-committed' message instead of a confusing negative number.
    """
    avail = total - used
    if unit:
        if avail < 0:
            return "Over-committed by %d %s" % (abs(int(avail)), unit)
        return "%d %s" % (int(avail), unit)
    if avail < 0:
        return "Over-committed by %s" % _size_or_dash(abs(avail))
    return _size_or_dash(avail)


def _cpu_flag(used: int, total: int) -> str:
    if total <= 0:
        return "-"
    if used > total:
        return "Over-committed"
    return "OK"


def _mem_flag(used: float, total: float) -> str:
    if total <= 0:
        return "-"
    if used > total:
        return "Over-committed"
    return "OK"


def _host_vcpus(h: Dict[str, Any]) -> int:
    """Host vCPU capacity = sockets * cores * threads.

    oVirt presents a host's CPU topology as sockets / cores (per socket) /
    threads (hyper-threading). Counting threads gives the real vCPU capacity,
    e.g. 2 sockets * 28 cores * 2 threads = 112 vCPU per host.
    """
    top = h.get("cpu", {}).get("topology", {}) or {}
    sockets = _to_int(_leaf(top, "sockets"))
    cores = _to_int(_leaf(top, "cores"))
    threads = _to_int(_leaf(top, "threads")) or 1
    return sockets * cores * threads


class Report:
    """Builds the final PDF document."""

    def __init__(self, engine: str, username: str, data: Dict[str, Any],
                 name: str = "", version: str = "", platform: str = "",
                 logo: str = ""):
        self.engine = engine
        self.username = username
        self.data = data or {}
        self.generated_by = name or username or "N/A"
        self.engine_version = version or str(self.data.get("engine_version") or "") or "N/A"
        self.platform = platform.strip() or "Oracle Linux Virtualization Manager"
        self.logo = logo.strip()
        self.styles: Dict[str, ParagraphStyle] = {}
        self.story: List[Any] = []
        self.disk_by_vm: Dict[str, List[Dict[str, Any]]] = {}
        self.disk_by_id: Dict[str, Dict[str, Any]] = {}
        self.sd_by_id: Dict[str, str] = {}
        self.profile_to_net: Dict[str, str] = {}
        self.net_map: Dict[str, Dict[str, str]] = {}
        self.dc_map: Dict[str, str] = {}
        self.cluster_map: Dict[str, str] = {}
        self.host_nics_by_host: Dict[str, List[Dict[str, Any]]] = {}
        self.host_by_id: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Styles
    # ------------------------------------------------------------------ #
    def _build_styles(self):
        base = getSampleStyleSheet()
        self.styles = {
            "H1": ParagraphStyle(
                "H1", parent=base["Heading1"], fontSize=15, textColor=HEADER_BG,
                spaceBefore=14, spaceAfter=8,
            ),
            "H2": ParagraphStyle(
                "H2", parent=base["Heading2"], fontSize=11.5, textColor=SUBHEADER_BG,
                spaceBefore=8, spaceAfter=5,
            ),
            "Body": ParagraphStyle(
                "Body", parent=base["BodyText"], fontSize=9, leading=12,
            ),
            "Cell": ParagraphStyle(
                "Cell", parent=base["BodyText"], fontSize=9, leading=12,
                alignment=TA_JUSTIFY,
            ),
            "CellLeft": ParagraphStyle(
                "CellLeft", parent=base["BodyText"], fontSize=9, leading=12,
                alignment=TA_LEFT,
            ),
            "CellHeader": ParagraphStyle(
                "CellHeader", parent=base["BodyText"], fontSize=9.5, leading=12,
                textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER,
            ),
        }

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #
    # Columns are scaled so the table fills the full usable page width, with
    # both edges aligned to the margins.
    FULL_WIDTH = landscape(A4)[0] - 2 * 15 * mm

    def _table(self, headers: List[str], rows: List[List[Any]],
               col_widths: Optional[List[float]] = None,
               left_cols: Optional[set] = None,
               block_bg: Optional[List[tuple]] = None):
        left_cols = left_cols or set()
        data = [[Paragraph(h, self.styles["CellHeader"]) for h in headers]]
        for ri, r in enumerate(rows):
            data.append([Paragraph(_txt(c).replace("&", "&amp;").replace("<", "&lt;")
                                   .replace(">", "&gt;").replace("\n", "<br/>"),
                                   self.styles["CellLeft"] if ci in left_cols else self.styles["Cell"])
                         for ci, c in enumerate(r)])
        if col_widths:
            total = sum(col_widths)
            if total:
                col_widths = [w * self.FULL_WIDTH / total for w in col_widths]
        t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b6c6d8")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        if block_bg:
            # Uniform alternating background per block (used for merged rows)
            for bi, (s, e) in enumerate(block_bg):
                bg = ALT_ROW if bi % 2 else colors.white
                style.append(("BACKGROUND", (0, s), (-1, e), bg))
        else:
            for i in range(1, len(data)):
                if i % 2 == 0:
                    style.append(("BACKGROUND", (0, i), (-1, i), ALT_ROW))
        t.setStyle(TableStyle(style))
        return t

    def _heading(self, text: str) -> Paragraph:
        return Paragraph(text, self.styles["H1"])

    def _subheading(self, text: str) -> Paragraph:
        return Paragraph(text, self.styles["H2"])

    def _meta_block(self):
        """Label/value meta block on the first page with column-aligned values."""
        now = datetime.now()
        label_style = ParagraphStyle(
            "MetaLabel", parent=self.styles["Body"], fontSize=9,
            textColor=colors.HexColor("#6b7b8c"), leading=13, spaceAfter=0,
        )
        value_style = ParagraphStyle(
            "MetaValue", parent=self.styles["Body"], fontSize=9.5,
            textColor=colors.HexColor("#0b2c47"), leading=13, spaceAfter=0,
        )
        label_width = 38 * mm
        meta_rows = [
            ("ENGINE DOMAIN", self.engine),
            ("ENGINE VERSION", self.engine_version),
            ("PLATFORM", self.platform),
            ("USER", self.username),
            ("GENERATED", self.generated_by),
            ("REPORT DATE", now.strftime("%Y-%m-%d")),
            ("REPORT TIME", now.strftime("%H:%M:%S")),
        ]
        data = [[Paragraph(label, label_style), Paragraph(value, value_style)]
                for label, value in meta_rows]
        t = Table(data, colWidths=[label_width, self.FULL_WIDTH - label_width])
        t.hAlign = "LEFT"
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 6))

    # ------------------------------------------------------------------ #
    # Sections
    # ------------------------------------------------------------------ #
    def _summary(self):
        self.story.append(self._heading("Summary"))
        counts = {
            "Datacenters": len(_lists(self.data, "datacenters")),
            "Clusters": len(_lists(self.data, "clusters")),
            "Hosts": len(_lists(self.data, "hosts")),
            "Storage Domains": len(_lists(self.data, "storagedomains")),
            "Networks": len(_lists(self.data, "networks")),
            "Virtual Machines": len(_lists(self.data, "vms")),
        }
        rows = []
        keys = list(counts.keys())
        for i in range(0, len(keys), 2):
            row = []
            for k in keys[i:i + 2]:
                row += [k, counts[k]]
            rows.append(row)
        self.story.append(self._table(["Component", "Count", "Component", "Count"],
                                      rows, col_widths=[120, 50, 120, 50]))

    def _total_utilization(self):
        self.story.append(self._heading("Total Utilization"))

        # CPU: total = sum of host vCPU capacity; used = sum of allocated VM vCPUs
        total_cores = sum(_host_vcpus(h) for h in _lists(self.data, "hosts"))
        used_cores = 0
        for vm in _lists(self.data, "vms"):
            top = vm.get("cpu", {}).get("topology", {}) or {}
            sockets = _to_int(_leaf(top, "sockets"))
            cores = _to_int(_leaf(top, "cores"))
            used_cores += sockets * cores

        # Memory: total = sum of host memory; used = sum of allocated VM memory
        total_mem = sum(_to_int(_leaf(h, "memory")) for h in _lists(self.data, "hosts"))
        used_mem = sum(_to_int(_leaf(vm, "memory")) for vm in _lists(self.data, "vms"))

        # Storage: sum of storage-domain used / available
        used_storage = 0
        avail_storage = 0
        for sd in _lists(self.data, "storagedomains"):
            used_storage += _to_int(_leaf(sd, "used"))
            avail_storage += _to_int(_leaf(sd, "available"))

        rows = [
            ["CPU",
             "%d vCPU" % used_cores,
             _avail_display(used_cores, total_cores, "vCPU")],
            ["Memory",
             _size_or_dash(used_mem),
             _avail_display(used_mem, total_mem)],
            ["Storage",
             _size_or_dash(used_storage),
             _size_or_dash(avail_storage)],
        ]
        self.story.append(self._table(
            ["Resource", "Used", "Available"],
            rows, col_widths=[110, 110, 110],
            left_cols={1, 2}))

    def _per_host_utilization(self):
        self.story.append(self._heading("Per-Host Utilization"))
        hosts = _lists(self.data, "hosts")
        if not hosts:
            self.story.append(Paragraph("No hosts found.", self.styles["Body"]))
            return

        # group allocated VM cores/memory by the host each VM runs on
        used_by_host: Dict[str, Dict[str, int]] = {}
        for vm in _lists(self.data, "vms"):
            top = vm.get("cpu", {}).get("topology", {}) or {}
            sockets = _to_int(_leaf(top, "sockets"))
            cores = _to_int(_leaf(top, "cores"))
            used_cores = sockets * cores
            used_mem = _to_int(_leaf(vm, "memory"))
            hid = _leaf(vm, "host", "id")
            if hid and hid in self.host_by_id:
                b = used_by_host.setdefault(hid, {"cores": 0, "mem": 0})
                b["cores"] += used_cores
                b["mem"] += used_mem

        rows = []
        for h in hosts:
            hid = _leaf(h, "id")
            hname = _leaf(h, "name") or hid or "-"
            total_cores = _host_vcpus(h)
            total_mem = _to_int(_leaf(h, "memory"))

            alloc = used_by_host.get(hid, {"cores": 0, "mem": 0})
            used_cores = alloc["cores"]
            used_mem = alloc["mem"]

            rows.append([
                hname,
                "%d / %d vCPU" % (used_cores, total_cores),
                _cpu_flag(used_cores, total_cores),
                "%s / %s" % (_size_or_dash(used_mem), _size_or_dash(total_mem)),
                _mem_flag(used_mem, total_mem),
            ])

        self.story.append(self._table(
            ["Host", "CPU (Used/Total)", "CPU Status", "Memory (Used/Total)", "Memory Status"],
            rows, col_widths=[140, 90, 80, 110, 90],
            left_cols={0, 1, 3}))

    def _datacenters(self):
        self.story.append(self._heading("Datacenters"))
        entry = _lists(self.data, "datacenters")
        if not entry:
            self.story.append(Paragraph("No datacenters found.", self.styles["Body"]))
            return
        rows = [[_leaf(dc, "name"), _leaf(dc, "id"), _status(_leaf(dc, "status")),
                 _leaf(dc, "description")] for dc in entry]
        self.story.append(self._table(
            ["Name", "ID", "Status", "Description"], rows,
            col_widths=[110, 180, 60, 130]))

    def _clusters(self):
        self.story.append(self._heading("Clusters"))
        entry = _lists(self.data, "clusters")
        if not entry:
            self.story.append(Paragraph("No clusters found.", self.styles["Body"]))
            return
        rows = []
        for cl in entry:
            ver = _leaf(cl, "version", "major") + "." + _leaf(cl, "version", "minor")
            rows.append([
                _leaf(cl, "name"),
                self._dc_name(node=cl),
                _leaf(cl, "cpu", "architecture"),
                _leaf(cl, "cpu", "type"),
                ver,
                _leaf(cl, "description"),
            ])
        self.story.append(self._table(
            ["Cluster", "Datacenter", "CPU Architecture", "CPU Type",
             "Version", "Description"], rows,
            col_widths=[90, 95, 80, 120, 50, 80]))

    def _host_network_speed(self, host_id: str) -> str:
        """Return the fastest NIC speed for a host (e.g. '10000 Mbps')."""
        speeds = []
        for nic in self.host_nics_by_host.get(host_id, []):
            raw = _leaf(nic, "speed")
            try:
                bps = float(raw)
            except (TypeError, ValueError):
                continue
            if bps > 0:
                speeds.append(bps)
        if not speeds:
            return "-"
        mbps = max(speeds) / 1_000_000
        return f"{mbps:.0f} Mbps"

    def _hosts(self):
        self.story.append(self._heading("Hosts"))
        entry = _lists(self.data, "hosts")
        if not entry:
            self.story.append(Paragraph("No hosts found.", self.styles["Body"]))
            return
        rows = []
        for h in entry:
            mem = _fmt_size(_leaf(h, "memory")) if _leaf(h, "memory") else "-"
            sockets = _leaf(h, "cpu", "topology", "sockets")
            cores = _leaf(h, "cpu", "topology", "cores")
            cpu_model = _leaf(h, "cpu", "name") or _leaf(h, "cpu", "type") or "-"
            sockets_disp = sockets if sockets else "-"
            cores_disp = cores if cores else "-"
            mem_disp = mem or "-"
            status_disp = _status(_leaf(h, "status"))
            name_disp = _leaf(h, "name") or "-"
            rows.append([
                name_disp,
                status_disp,
                cpu_model,
                sockets_disp,
                cores_disp,
                mem_disp,
                self._host_network_speed(_leaf(h, "id")),
                _leaf(h, "hardware_information", "product_name") or "-",
                _leaf(h, "hardware_information", "serial_number") or "-",
            ])
        self.story.append(self._table(
            ["Host", "Status", "CPU Model", "Sockets", "Cores",
             "Memory", "Network Speed", "Product Name", "Serial Number"],
            rows, col_widths=[125, 45, 100, 46, 40, 55, 60, 85, 75],
            left_cols={0, 2, 6, 7}))

    def _storage(self):
        self.story.append(self._heading("Storage Domains"))
        entry = _lists(self.data, "storagedomains")
        if not entry:
            self.story.append(Paragraph("No storage domains found.", self.styles["Body"]))
            return
        rows = []
        for sd in entry:
            used = _fmt_size(_leaf(sd, "used")) if _leaf(sd, "used") else ""
            avail = _fmt_size(_leaf(sd, "available")) if _leaf(sd, "available") else ""
            # Datacenter comes from the data_centers link list (ref only)
            dc_nodes = _lists(sd.get("data_centers", {}), "data_center") + _lists(
                sd.get("data_centers", {}).get("storage_domain", {}), "storage_domain")
            dc_name = ", ".join(dict.fromkeys(self._dc_name(n) for n in dc_nodes
                                              if self._dc_name(n) != "-")) or "-"
            if not dc_nodes:
                dc_name = self._dc_name(node=sd)
            status = _status(_leaf(sd, "status"))
            rows.append([
                _leaf(sd, "name"),
                _leaf(sd, "type"),
                _leaf(sd, "storage", "type"),
                dc_name,
                status,
                used, avail,
            ])
        self.story.append(self._table(
            ["Storage Domain", "Domain Type", "Storage Type", "Datacenter", "Status",
             "Used", "Available"],
            rows, col_widths=[95, 65, 70, 90, 50, 50, 50]))

    def _networks(self):
        self.story.append(self._heading("Networks"))
        entry = _lists(self.data, "networks")
        if not entry:
            self.story.append(Paragraph("No networks found.", self.styles["Body"]))
            return
        rows = []
        for nw in entry:
            vlan = _leaf(nw, "vlan", "id")
            if not vlan or vlan == "0":
                vlan_disp = "Untagged"
            else:
                vlan_disp = vlan
            mtu = _leaf(nw, "mtu")
            if not mtu or mtu == "0":
                mtu_disp = "1500 (default)"
            else:
                mtu_disp = mtu
            rows.append([
                _leaf(nw, "name"),
                vlan_disp,
                mtu_disp,
                self._dc_name(node=nw),
            ])
        self.story.append(self._table(
            ["Network", "VLAN", "MTU", "Datacenter"],
            rows, col_widths=[160, 70, 90, 120]))

    def _vms(self):
        self.story.append(self._heading("Virtual Machines"))
        entry = _lists(self.data, "vms")
        if not entry:
            self.story.append(Paragraph("No virtual machines found.", self.styles["Body"]))
            return

        headers = ["VM Name", "Host", "Status", "OS Type", "CPU", "Memory",
                   "Network", "Storage"]
        rows = []
        for vm in entry:
            vid = _leaf(vm, "id")
            vm_name = _leaf(vm, "name") or vid
            host_ref = vm.get("host", {}) or {}
            host_id = _leaf(host_ref, "id")
            if not host_id:
                host_id = self._id_from_href(_leaf(host_ref, "href"))
            host = _name(self.host_by_id.get(host_id)) if self.host_by_id.get(host_id) \
                else (_name(host_ref) or "-")
            status = _status(_leaf(vm, "status"))
            mem = _fmt_size(_leaf(vm, "memory")) if _leaf(vm, "memory") else ""
            sockets = _leaf(vm, "cpu", "topology", "sockets")
            cores = _leaf(vm, "cpu", "topology", "cores")
            cpu = f"{cores} vCPU" if cores else ""
            if sockets and cores:
                cpu = f"{sockets}S/{cores}C"
            os_type = _leaf(vm, "os", "type") or self._readable_os(_leaf(vm, "guest_operating_system", "codename"))

            # Networks: interface (network name [VLAN]) [ip]
            mac_devs = self._mac_guest_devs(vm)
            networks = []
            for nic in _lists(vm.get("nics", {}), "nic"):
                nic_name = _leaf(nic, "name")
                net_name = self._network_name_for_nic(nic)
                vlan = self._vlan_for_nic(nic)
                nic_mac = _leaf(nic, "mac", "address")
                dev = mac_devs.get(nic_mac, {})
                if_name = _leaf(dev, "name") or nic_name
                label = f"{if_name}: {net_name}" if net_name else if_name
                if vlan:
                    label += f" (VLAN {vlan})"
                ips = dev.get("ips") or []
                if ips:
                    label += " [" + " ".join(ips) + "]"
                networks.append(label)
            net_text = "\n".join(networks) or "-"

            # Storage: one cell, each disk on its own line (name [size])
            disk_list = self.disk_by_vm.get(vid) or []
            disk_lines = []
            for d in disk_list:
                disk_name, size, _sd = self._disk_details(d)
                part = disk_name
                if size:
                    part += f"  [{size}]"
                disk_lines.append(part)
            disk_text = "\n".join(disk_lines) if disk_lines else "-"

            rows.append([
                vm_name, host, status, os_type, cpu, mem, net_text, disk_text,
            ])

        self.story.append(self._table(
            headers, rows, col_widths=[78, 82, 42, 88, 60, 75, 120, 130],
            left_cols={6, 7}))

    @staticmethod
    def _readable_os(codename: str) -> str:
        return codename if codename else ""

    @staticmethod
    def _id_from_href(href: str) -> str:
        """Extract the trailing resource id from an oVirt href.

        e.g. '/ovirt-engine/api/hosts/5d820ba6-.../statistics' -> '5d820ba6-...'
        Returns '' when no id can be found.
        """
        seg = str(href).strip("/")
        parts = [p for p in seg.split("/") if p]
        for i, p in enumerate(parts):
            if p == "hosts" and i + 1 < len(parts):
                return parts[i + 1]
        return ""

    def _guest_ip_list(self, vm: Dict[str, Any]) -> List[List[str]]:
        """Return [[interface, ip], ...] for a VM."""
        out = []
        for dev in _lists(vm.get("reported_devices", {}), "reported_device"):
            name = _leaf(dev, "name")
            for ip in _lists(dev.get("ips", {}), "ip"):
                address = _leaf(ip, "address")
                if address:
                    out.append([name or "", address])
        return out

    def _mac_guest_devs(self, vm: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Return {mac: {"name": iface, "ips": [ip, ...]}} for a VM."""
        mac_devs: Dict[str, Dict[str, Any]] = {}
        for dev in _lists(vm.get("reported_devices", {}), "reported_device"):
            mac = _leaf(dev, "mac", "address")
            if not mac:
                continue
            entry = {"name": _leaf(dev, "name"), "ips": []}
            for ip in _lists(dev.get("ips", {}), "ip"):
                address = _leaf(ip, "address")
                if address:
                    entry["ips"].append(address)
            mac_devs[mac] = entry
        return mac_devs

    def _disk_storage_name(self, att: Dict[str, Any]) -> str:
        sd = att.get("disk", {}).get("storage_domains", {})
        doms = _lists(sd, "storage_domain")
        return ", ".join(_name(d) for d in doms) or "-"

    def _guest_ips(self, vm: Dict[str, Any]) -> List[List[str]]:
        ips = []
        for dev in _lists(vm.get("reported_devices", {}), "reported_device"):
            name = _leaf(dev, "name")
            for ip in _lists(dev.get("ips", {}), "ip"):
                address = _leaf(ip, "address")
                if address:
                    ips.append([address, name])
        return ips

    # ------------------------------------------------------------------ #
    # Lookup maps (disk, storage domain, vnic profile -> network)
    # ------------------------------------------------------------------ #
    def _build_maps(self):
        # datacenter id -> name
        for dc in _lists(self.data, "datacenters"):
            self.dc_map[_leaf(dc, "id")] = _leaf(dc, "name")

        # cluster id -> name
        for cl in _lists(self.data, "clusters"):
            self.cluster_map[_leaf(cl, "id")] = _leaf(cl, "name")

        # host id -> host record
        for h in _lists(self.data, "hosts"):
            self.host_by_id[_leaf(h, "id")] = h

        # network id -> {name, vlan}
        self.net_map = {}
        for nw in _lists(self.data, "networks"):
            nid = _leaf(nw, "id")
            self.net_map[nid] = {"name": _leaf(nw, "name"), "vlan": _leaf(nw, "vlan", "id")}

        # storage domain id -> name
        for sd in _lists(self.data, "storagedomains"):
            self.sd_by_id[_leaf(sd, "id")] = _leaf(sd, "name")

        # disk id -> disk record (name, provisioned_size, storage_domains)
        for dsk in _lists(self.data, "disks"):
            self.disk_by_id[_leaf(dsk, "id")] = dsk

        # vnic profile id -> network id
        for vp in _lists(self.data, "vnicprofiles"):
            self.profile_to_net[_leaf(vp, "id")] = _leaf(vp.get("network", {}), "id")

        # link each VM to its disk attachments
        atts = self.data.get("diskattachments") or []
        for att in atts:
            vid = _leaf(att, "vm", "id")
            if vid:
                self.disk_by_vm.setdefault(vid, []).append(att)

        # link each host to its NICs
        host_nics = self.data.get("host_nics") or []
        for hd in host_nics:
            hid = _leaf(hd, "host", "id")
            if hid:
                self.host_nics_by_host.setdefault(hid, []).extend(
                    _lists(hd.get("nics", {}), "nic"))

    def _dc_name(self, ref: Optional[Dict[str, Any]] = None,
                 node: Optional[Dict[str, Any]] = None) -> str:
        """Resolve a datacenter reference/name to its display name."""
        if ref is None and node is not None:
            ref = node.get("data_center", {}) or {}
        nm = _name(ref) if ref else ""
        if not nm and ref:
            nm = self.dc_map.get(_leaf(ref, "id"), "")
        return nm or "-"

    def _cluster_name(self, ref: Optional[Dict[str, Any]] = None,
                      node: Optional[Dict[str, Any]] = None) -> str:
        """Resolve a cluster reference/name to its display name."""
        if ref is None and node is not None:
            ref = node.get("cluster", {}) or {}
        nm = _name(ref) if ref else ""
        if not nm and ref:
            nm = self.cluster_map.get(_leaf(ref, "id"), "")
        return nm or "-"

    def _nic_network_name(self, n: Dict[str, Any]) -> str:
        """Resolve a host NIC's bound network id to its display name."""
        net = n.get("network", {}) or {}
        if net:
            return _name(net) or self.net_map.get(_leaf(net, "id"), {}).get("name", "-")
        return "-"

    def _network_name_for_nic(self, nic: Dict[str, Any]) -> str:
        """Resolve the network name for a NIC via its vnic profile."""
        profile_id = _leaf(nic, "vnic_profile", "id") or _leaf((nic.get("vnic_profile") or {}), "id")
        net_id = self.profile_to_net.get(profile_id, "")
        if net_id and net_id in self.net_map:
            return self.net_map[net_id]["name"]
        # fall back to any inline network name
        inline = _name((nic.get("vnic_profile") or {}).get("network", {}))
        return inline or ""

    def _vlan_for_nic(self, nic: Dict[str, Any]) -> str:
        profile_id = _leaf(nic, "vnic_profile", "id") or _leaf((nic.get("vnic_profile") or {}), "id")
        net_id = self.profile_to_net.get(profile_id, "")
        if net_id and net_id in self.net_map:
            return self.net_map[net_id]["vlan"]
        return ""

    def _disk_details(self, att: Dict[str, Any]):
        """Return (name, size_str, storage_domain_names) for a disk attachment."""
        disk = att.get("disk", {}) or {}
        disk_id = _leaf(disk, "id")
        record = self.disk_by_id.get(disk_id, disk)

        name = _leaf(record, "name") or _leaf(record, "alias") or "-"
        size = _leaf(record, "provisioned_size") or _leaf(record, "size")
        size_str = _fmt_size(size) if size else ""

        # storage domains on the disk record
        sds = _lists(record.get("storage_domains", {}), "storage_domain")
        names = []
        for sd in sds:
            nm = _name(sd) or self.sd_by_id.get(_leaf(sd, "id"), "")
            if nm:
                names.append(nm)
        return name, size_str, ", ".join(names) or "-"

    def _disk_storage_name(self, att: Dict[str, Any]) -> str:
        _, _, sdn = self._disk_details(att)
        return sdn

    # ------------------------------------------------------------------ #
    # Cover + assemble
    # ------------------------------------------------------------------ #
    def _counts(self):
        return {
            "Datacenters": len(_lists(self.data, "datacenters")),
            "Clusters": len(_lists(self.data, "clusters")),
            "Hosts": len(_lists(self.data, "hosts")),
            "Storage Domains": len(_lists(self.data, "storagedomains")),
            "Networks": len(_lists(self.data, "networks")),
            "Virtual Machines": len(_lists(self.data, "vms")),
        }

    def _cover_flowables(self, pagesize) -> List[Any]:
        # Cover content is drawn directly on the canvas in _draw_cover_bg for
        # precise positioning (no overlap). Only a tiny spacer is used here so
        # the page is emitted.
        return [Spacer(1, 1)]

    def _draw_cover_bg(self, canvas, pagesize):
        """Draw the complete cover page directly on the canvas."""
        w, h = pagesize
        counts = self._counts()
        now = datetime.now()
        labels = [("Datacenters", counts["Datacenters"]), ("Clusters", counts["Clusters"]),
                  ("Hosts", counts["Hosts"]), ("Storage Domains", counts["Storage Domains"]),
                  ("Networks", counts["Networks"]), ("VMs", counts["Virtual Machines"])]

        c = canvas
        c.saveState()

        # ---- Background ----
        c.setFillColor(colors.HexColor("#f4f7fb"))
        c.rect(0, 0, w, h, stroke=0, fill=1)

        # ---- Top header band ----
        band_h = 155
        c.setFillColor(HEADER_BG)
        c.rect(0, h - band_h, w, band_h, stroke=0, fill=1)

        # decorative circles on the band (top-right)
        c.setFillColor(colors.HexColor("#dce6f2"))
        c.circle(w - 70, h - 75, 46, stroke=0, fill=0)
        c.setFillColor(colors.HexColor("#2f6ea8"))
        c.circle(w - 44, h - 46, 22, stroke=0, fill=1)

        # ---- Title inside the band ----
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 30)
        c.drawCentredString(w / 2, h - 92, "INVENTORY REPORT")
        c.setFont("Helvetica", 13)
        c.setFillColor(colors.HexColor("#dceafa"))
        c.drawCentredString(w / 2, h - 56,
                            f"{self.platform}  |  Infrastructure Overview")

        # ---- Stat tiles (3x2) ----
        tile_w, tile_h = 205, 52
        gap = 18
        grid_w = 3 * tile_w + 2 * gap
        x0 = (w - grid_w) / 2
        start_y = h - band_h - 78   # below band
        colors_tile = [colors.HexColor("#eef3fa")] * 6
        for idx, (label, value) in enumerate(labels):
            col = idx % 3
            row = idx // 3
            x = x0 + col * (tile_w + gap)
            y = start_y - row * (tile_h + 16)
            # tile background
            c.setFillColor(colors.HexColor("#eef3fa"))
            c.roundRect(x, y, tile_w, tile_h, 6, stroke=0, fill=1)
            c.setStrokeColor(colors.HexColor("#c6d4e4"))
            c.setLineWidth(0.7)
            c.roundRect(x, y, tile_w, tile_h, 6, stroke=1, fill=0)
            # number and label
            c.setFillColor(colors.HexColor("#1f4e78"))
            c.setFont("Helvetica-Bold", 22)
            c.drawCentredString(x + tile_w / 2, y + tile_h - 26, str(value))
            c.setFillColor(colors.HexColor("#6b7b8c"))
            c.setFont("Helvetica", 9)
            c.drawCentredString(x + tile_w / 2, y + 10, label)

        # ---- Meta block (bordered label/value table) ----
        meta_rows = [
            ("ENGINE DOMAIN", self.engine),
            ("ENGINE VERSION", self.engine_version),
            ("PLATFORM", self.platform),
            ("USER", self.username),
            ("GENERATED", self.generated_by),
            ("REPORT DATE", now.strftime("%Y-%m-%d")),
            ("REPORT TIME", now.strftime("%H:%M:%S")),
        ]
        row_h = 26
        col_label = 170
        table_w = 400
        table_h = row_h * len(meta_rows)
        top = start_y - 2 * (tile_h + 16) - 60
        left = (w - table_w) / 2
        # table background + border (matches the stat-tile style)
        c.setFillColor(colors.HexColor("#f4f7fb"))
        c.roundRect(left, top - table_h, table_w, table_h, 6, stroke=0, fill=1)
        c.setStrokeColor(colors.HexColor("#c6d4e4"))
        c.setLineWidth(0.7)
        c.roundRect(left, top - table_h, table_w, table_h, 6, stroke=1, fill=0)
        for i, (label, value) in enumerate(meta_rows):
            yy = top - i * row_h
            if i > 0:
                c.setStrokeColor(colors.HexColor("#dfe7f0"))
                c.setLineWidth(0.5)
                c.line(left + 8, yy, left + table_w - 8, yy)
            # label cell (right-aligned)
            c.setFillColor(colors.HexColor("#6b7b8c"))
            c.setFont("Helvetica-Bold", 9.5)
            c.drawRightString(left + col_label - 10, yy - row_h + 9, label)
            # value cell (left-aligned)
            c.setFillColor(colors.HexColor("#0b2c47"))
            c.setFont("Helvetica", 11)
            c.drawString(left + col_label - 2, yy - row_h + 9, value)

        # ---- Bottom accent ----
        c.setFillColor(ACCENT)
        c.rect(0, 0, w, 6, stroke=0, fill=1)

        c.restoreState()

    def build(self, output_path: str) -> str:
        self._build_styles()
        self._build_maps()

        pagesize = landscape(A4)

        self.story = []
        self._meta_block()
        self._summary()
        self._total_utilization()
        self._per_host_utilization()
        self._datacenters()
        self._clusters()
        self._hosts()
        self._storage()
        self._networks()
        self._vms()

        doc = BaseDocTemplate(
            output_path,
            pagesize=pagesize,
            leftMargin=15 * mm, rightMargin=15 * mm,
            topMargin=27 * mm, bottomMargin=16 * mm,
            title=f"Inventory Report - {self.engine}",
            author=self.username,
        )

        HEADER_RULE = colors.HexColor("#b6c6d8")
        total_pages = 0

        def header_footer(canvas, d):
            canvas.saveState()
            H = pagesize[1]
            W = pagesize[0]
            # ---- Logo (right, above the rule) ----
            logo_h = 16 * mm
            logo_bottom = H - 21 * mm
            logo_w = 16 * mm
            if self.logo:
                from PIL import Image as _PILImage
                from reportlab.lib.utils import ImageReader
                try:
                    _img = _PILImage.open(self.logo).convert("RGBA")
                    _ratio = _img.size[0] / max(_img.size[1], 1)
                    logo_w = max(8 * mm, min(logo_h * _ratio, 55 * mm))
                    _img.load()
                    _reader = ImageReader(_img)
                    canvas.drawImage(_reader, W - 15 * mm - logo_w, logo_bottom,
                                     logo_w, logo_h,
                                     mask="auto", preserveAspectRatio=True,
                                     anchor="c")
                except Exception:
                    pass
            # ---- Title (left, bold) ----
            canvas.setFillColor(colors.HexColor("#0b2c47"))
            canvas.setFont("Helvetica-Bold", 16)
            canvas.drawString(15 * mm, H - 13 * mm,
                              "Infrastructure Report")
            # ---- Subtitle (below the title, left) ----
            canvas.setFillColor(colors.HexColor("#556b82"))
            canvas.setFont("Helvetica", 8)
            canvas.drawString(15 * mm, H - 20 * mm,
                              f"{self.platform}  |  {self.engine}")
            # ---- Horizontal rule below the header ----
            canvas.setStrokeColor(HEADER_RULE)
            canvas.setLineWidth(1.2)
            canvas.line(15 * mm, H - 24 * mm, W - 15 * mm, H - 24 * mm)
            # ---- Footer ----
            canvas.setStrokeColor(colors.HexColor("#c6d4e4"))
            canvas.setLineWidth(0.5)
            canvas.line(15 * mm, 10 * mm, pagesize[0] - 15 * mm, 10 * mm)
            canvas.setFillColor(colors.HexColor("#8893a0"))
            canvas.setFont("Helvetica", 7.5)
            canvas.drawString(15 * mm, 6.5 * mm,
                              f"Generated By {self.generated_by}")
            canvas.setFillColor(colors.HexColor("#8893a0"))
            canvas.setFont("Helvetica", 7.5)
            canvas.drawRightString(pagesize[0] - 15 * mm, 6.5 * mm,
                                   f"Page {d.page}/{total_pages}")
            canvas.restoreState()

        main_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
        doc.addPageTemplates([
            PageTemplate(id="main", frames=[main_frame], onPage=header_footer),
        ])

        # First pass: count total pages so the footer can show "Page X/Y".
        import io as _io
        _count_buf = _io.BytesIO()
        _count_doc = BaseDocTemplate(
            _count_buf,
            pagesize=pagesize,
            leftMargin=doc.leftMargin, rightMargin=doc.rightMargin,
            topMargin=doc.topMargin, bottomMargin=doc.bottomMargin,
        )
        _count_doc.addPageTemplates([
            PageTemplate(id="count", frames=[
                Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="count")]),
        ])
        _count_doc.build(self.story)
        total_pages = max(_count_doc.page, 1)

        # Second pass: render the real document with the correct total.
        self.story = []
        self._meta_block()
        self._summary()
        self._total_utilization()
        self._per_host_utilization()
        self._datacenters()
        self._clusters()
        self._hosts()
        self._storage()
        self._networks()
        self._vms()

        doc.build(self.story)
        return output_path


class _ArgumentParser(argparse.ArgumentParser):
    """Parser whose Usage line shows the current file name and a simplified error."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("prog", self._prog_name())
        super().__init__(*args, **kwargs)

    def _script(self) -> str:
        name = os.path.basename(sys.argv[0] or "") or "OLVMReportGenerator.py"
        if name.lower().endswith(".exe"):
            return name
        if name.lower().endswith(".py"):
            return name
        return name + ".py"

    def _prog_name(self) -> str:
        return self._script()

    def _usage_text(self) -> str:
        header = "Usage: " + self._prog_name()
        fmt = argparse.HelpFormatter(prog=self.prog, width=10000)
        fmt.add_usage(None, self._actions, self._mutually_exclusive_groups)
        oneline = fmt.format_help().strip()
        prefix = "usage: " + self.prog + " "
        options = oneline[len(prefix):] if oneline.startswith(prefix) else oneline

        words = options.split()
        tokens = []
        i = 0
        while i < len(words):
            w = words[i]
            joins_value = (w.startswith("-") and not w.startswith("[")) or \
                          (w.startswith("[") and not w.endswith("]"))
            if joins_value and i + 1 < len(words) \
                    and not words[i + 1].startswith("-") and not words[i + 1].startswith("["):
                tokens.append(w + " " + words[i + 1])
                i += 2
            else:
                tokens.append(w)
                i += 1

        width = 82
        opt_lines = []
        cur = ""
        for t in tokens:
            if not cur:
                cur = t
            elif len(cur) + 1 + len(t) <= width:
                cur += " " + t
            else:
                opt_lines.append(cur)
                cur = t
        if cur:
            opt_lines.append(cur)

        first_line = header + " " + opt_lines[0]
        indent = " " * (len(header) + 1)
        lines = [first_line]
        for l in opt_lines[1:]:
            lines.append(indent + l)
        return "\n".join(lines) + "\n"

    def error(self, message):
        import sys as _sys
        _sys.stderr.write(self._usage_text())
        self.exit(2, "Error! Arguments are required.\n")

    def format_help(self):
        text = super().format_help()
        idx = text.find("usage: ")
        sep = text.find("\n\n", idx)
        if idx != -1 and sep != -1:
            text = self._usage_text() + text[sep + 1:]
        return text


class _HelpFormatter(argparse.HelpFormatter):
    def format_help(self):
        return super().format_help().replace("options:", "Options:")


def prompt_required(prompt: str, value: str = "", secret: bool = False) -> str:
    """Return the provided value or interactively ask the user."""
    if value:
        return value
    if secret:
        return getpass.getpass(prompt)
    return input(prompt)


def main(argv=None) -> int:
    parser = _ArgumentParser(
        description="Generate a PDF inventory report from OLVM/oVirt Engine.",
        add_help=False,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help message and exit")
    parser.add_argument("--engine", required=True,
                        help="Engine domain, e.g. olvm.engine.local")
    parser.add_argument("--user", default="admin@internal",
                        help="Engine user (default: admin@internal)")
    parser.add_argument("--password", required=True,
                        help="Engine password (shown on the command line)")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: <engine>_report_<date>.pdf)")
    parser.add_argument("--insecure", action="store_true",
                        help="Skip SSL certificate verification (self-signed certs)")
    parser.add_argument("--name", required=True, metavar="NAME",
                        help="Name shown as 'Generated By' in the footer, "
                             "e.g. \"Aung Thu Myint\"")
    parser.add_argument("--platform", default="", metavar="NAME",
                        help="Platform display name (default: "
                             "\"Oracle Linux Virtualization Manager\"). "
                             "Use e.g. \"oVirt Engine\" for a plain oVirt engine.")
    parser.add_argument("--logo", required=True, metavar="FILE",
                        help="Logo image placed in the top-left of the "
                             "header (PNG recommended, e.g. 300x300 pixels).")
    parser.add_argument("--timeout", type=int, default=60,
                        help="HTTP timeout in seconds (default: 60)")
    args = parser.parse_args(argv)

    engine = prompt_required("Engine domain (e.g. olvm.engine.local): ", args.engine or "")
    username = args.user or "admin@internal"
    password = prompt_required("Password for {}: ".format(username), args.password or "", secret=True)

    if not engine:
        parser.error("An engine domain is required.")

    client = OLVMClient(
        engine=engine,
        username=username,
        password=password,
        insecure=args.insecure,
        timeout=args.timeout,
    )

    print(f"Connecting to {client.engine} ...")
    try:
        data = client.gather()
    except OLVMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    count_vms = len(data.get("vms") or [])
    if not count_vms:
        print("Connected, but no virtual machines were found.", file=sys.stderr)

    output = args.output or f"{engine.replace('://', '_').replace('/', '_')}_report_{_stamp()}.pdf"
    report = Report(engine=engine, username=username, data=data, name=args.name,
                    platform=args.platform, logo=args.logo)

    print(f"Report File Name [{output}]")
    try:
        report.build(output)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR while generating the PDF: {exc}", file=sys.stderr)
        return 1

    print("Report generated successfully.")
    return 0


def _stamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())

