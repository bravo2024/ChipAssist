"""data.py — Synthetic chip specification corpus for ChipAssist.

Semiconductor datasheet-style documents: pin tables, voltage specs,
timing parameters. Each query has ground-truth relevant docs and
expected extracted spec values.
"""
from __future__ import annotations
from typing import Any


def _doc(doc_id, part_name, category, content):
    return {"id": doc_id, "title": part_name, "category": category, "content": content}


def _q(qid, query, relevant, expected_spec=None):
    return {"id": qid, "query": query, "relevant_doc_ids": relevant,
            "expected_spec": expected_spec or {}}


DOCUMENTS = [
    _doc("CH-001", "MARVELL 88E1510 PHY", "ethernet",
         "The Marvell 88E1510 is a single-port Gigabit Ethernet PHY. Supply voltage is 3.3V for analog and 1.0V for core logic. It supports RGMII and SGMII MAC interfaces. The device has 48 pins in a QFN package. Maximum clock frequency is 125 MHz. Operating temperature range is -40 to 85 degrees Celsius."),
    _doc("CH-002", "MARVELL 88E6390 Switch", "ethernet",
         "The Marvell 88E6390 is a 11-port Gigabit Ethernet switch with integrated PHYs. Supply voltage is 1.0V core and 3.3V I/O. It supports 10M 100M and 1000M speeds. The device has 320 pins in a BGA package. Maximum switching capacity is 11 Gbps."),
    _doc("CH-003", "MARVELL Prestera DX3230", "switch",
         "The Prestera DX3230 is a 24-port 10G Ethernet switch. Core voltage is 1.0V and I/O voltage is 1.8V. The device supports L2 and L3 switching with 480 Gbps switching capacity. It has 784 pins in a BGA package. Maximum operating frequency is 156.25 MHz for XAUI interfaces."),
    _doc("CH-004", "MARVELL Armada 3720", "processor",
         "The Armada 3720 is a dual-core ARM Cortex-A72 processor running at 1.0 GHz. Core voltage is 0.9V and I/O voltage is 1.8V. It has 620 pins in a FCBGA package. The processor includes a hardware crypto engine and supports DDR4 at 1600 MHz. L2 cache size is 1 MB."),
    _doc("CH-005", "MARVELL OCTEON TX2 CN73XX", "processor",
         "The OCTEON TX2 CN73XX is a 24-core MIPS64 processor. Core voltage is 0.9V and DDR voltage is 1.2V. It runs at 1.5 GHz with 784 pins in a BGA package. L2 cache is 2 MB shared. The processor supports DDR4 at 2400 MHz and has integrated Ethernet MAC."),
    _doc("CH-006", "MARVELL NVMe Controller 88SN2401", "storage",
         "The 88SN2401 is an NVMe over Fabrics controller. Core voltage is 0.9V and I/O voltage is 1.2V. It supports PCIe Gen4 x4 with 16 lanes. The device has 256 pins in a FCBGA package. Maximum throughput is 32 Gbps per lane. It supports up to 8 NVMe namespaces."),
    _doc("CH-007", "MARVELL 88SE9230 SATA Controller", "storage",
         "The 88SE9230 is a 4-port SATA 6Gbps controller. Core voltage is 1.2V and I/O voltage is 3.3V. It uses PCIe Gen2 x2 interface. The device has 132 pins in a TQFP package. Maximum SATA speed is 6 Gbps per port. It supports RAID 0 1 and 10."),
    _doc("CH-008", "MARVELL 88W8997 WiFi", "wireless",
         "The 88W8997 is a 2x2 MIMO WiFi 5 and Bluetooth 5 combo chip. Core voltage is 1.2V and I/O voltage is 1.8V. It supports 802.11 ac with max rate of 866 Mbps. The device has 168 pins in a WLCSP package. Operating frequency is 2.4 GHz and 5 GHz."),
    _doc("CH-009", "MARVELL 88W9098 WiFi 6", "wireless",
         "The 88W9098 is a 4x4 MIMO WiFi 6 and Bluetooth 5 combo chip. Core voltage is 0.9V and I/O voltage is 1.8V. It supports 802.11 ax with max rate of 4.8 Gbps. The device has 240 pins in a FCBGA package. It supports OFDMA and MU-MIMO with up to 1024-QAM."),
    _doc("CH-010", "MARVELL SEC 5 Security Engine", "security",
         "The SEC 5 is a hardware security engine. Core voltage is 0.9V. It supports AES-128 and AES-256 encryption with 10 Gbps throughput. RSA key sizes up to 4096 bits are supported. SHA-256 and SHA-512 are available. The engine includes a true random number generator."),
]


QUERIES = [
    _q("Q-01", "What is the supply voltage of the 88E1510 PHY?", ["CH-001"], {"voltage": "3.3V"}),
    _q("Q-02", "How many pins does the 88E6390 switch have?", ["CH-002"], {"pins": "320"}),
    _q("Q-03", "What is the switching capacity of the DX3230?", ["CH-003"], {"capacity": "480 Gbps"}),
    _q("Q-04", "What is the clock speed of the Armada 3720?", ["CH-004"], {"clock": "1.0 GHz"}),
    _q("Q-05", "How many cores does the OCTEON TX2 have?", ["CH-005"], {"cores": "24"}),
    _q("Q-06", "What PCIe generation does the 88SN2401 support?", ["CH-006"], {"pcie": "Gen4"}),
    _q("Q-07", "How many SATA ports does the 88SE9230 have?", ["CH-007"], {"ports": "4"}),
    _q("Q-08", "What WiFi standard does the 88W8997 support?", ["CH-008"], {"wifi": "ac"}),
    _q("Q-09", "What is the max rate of the 88W9098?", ["CH-009"], {"rate": "4.8 Gbps"}),
    _q("Q-10", "What encryption does the SEC 5 support?", ["CH-010"], {"encryption": "AES-128"}),
]


def make_corpus():
    return {
        "documents": DOCUMENTS,
        "queries": QUERIES,
        "n_documents": len(DOCUMENTS),
        "n_queries": len(QUERIES),
        "categories": sorted(set(d["category"] for d in DOCUMENTS)),
    }