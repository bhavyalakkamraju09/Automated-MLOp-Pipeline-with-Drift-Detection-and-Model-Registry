"""Unit tests for PSI threshold logic in the drift detector."""
from src.monitoring.drift_detector import check_alert_threshold


def test_no_alert_below_threshold():
    metrics = {"psi_tenure": 0.05, "psi_MonthlyCharges": 0.08}
    alert, critical = check_alert_threshold(metrics)
    assert not alert
    assert not critical


def test_alert_but_not_critical():
    metrics = {"psi_tenure": 0.22, "psi_MonthlyCharges": 0.05}
    alert, critical = check_alert_threshold(metrics)
    assert alert
    assert not critical


def test_critical_alert():
    metrics = {"psi_tenure": 0.30}
    alert, critical = check_alert_threshold(metrics)
    assert alert
    assert critical
