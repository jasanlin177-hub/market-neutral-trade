"""風控守門模組：訊號要進執行層前的不可繞過關卡。

設計原則：任何一項 BLOCK 級檢查未過，交易就不放行；
WARN 級檢查未過則放行但附帶警示（由使用者決定）。
"""
from dataclasses import dataclass, field
from enum import Enum


class CheckLevel(Enum):
    BLOCK = "block"   # 未過即擋下
    WARN = "warn"     # 未過僅警示


@dataclass
class CheckResult:
    name: str
    level: CheckLevel
    passed: bool
    detail: str


@dataclass
class GateDecision:
    approved: bool
    checks: list = field(default_factory=list)

    @property
    def blocks(self):
        return [c for c in self.checks if not c.passed and c.level == CheckLevel.BLOCK]

    @property
    def warnings(self):
        return [c for c in self.checks if not c.passed and c.level == CheckLevel.WARN]

    def summary(self) -> str:
        lines = ["=== 風控守門結果 ==="]
        for c in self.checks:
            mark = "PASS" if c.passed else ("BLOCK" if c.level == CheckLevel.BLOCK else "WARN")
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        lines.append(f"=> {'放行' if self.approved else '擋下，不得進入執行層'}")
        return "\n".join(lines)


def evaluate(checks: list) -> GateDecision:
    """彙總所有檢查結果，任何 BLOCK 未過即擋下。"""
    approved = all(c.passed for c in checks if c.level == CheckLevel.BLOCK)
    return GateDecision(approved=approved, checks=checks)
