"""Standard Logger module supporting tab/space delimited log style (BPJS Ketenagakerjaan Section 6.1.1)."""

import logging
import os
import socket
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

# WIB Timezone (UTC+7)
WIB_TZ = timezone(timedelta(hours=7))


def get_wib_now() -> datetime:
    """Return current datetime in WIB timezone."""
    return datetime.now(WIB_TZ)


def get_wib_timestamp() -> str:
    """Return current timestamp formatted as YYYY-MM-DD HH:mm:ss.SSS in WIB (UTC+7)."""
    now = get_wib_now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"


def get_wib_date_str() -> str:
    """Return current date formatted as ddmmyyyy (e.g. '20082026') in WIB (UTC+7)."""
    now = get_wib_now()
    return now.strftime("%d%m%Y")


def get_server_ip() -> str:
    """Get local server IP or hostname."""
    try:
        host_name = socket.gethostname()
        local_ip = socket.gethostbyname(host_name)
        return local_ip if local_ip else host_name
    except Exception:
        return "127.0.0.1"


class StandardLogger:
    """BPJS Ketenagakerjaan 6.1.1 System Log Generator in clean text line format."""

    def __init__(
        self,
        service_name: str = "svc-face-compare",
        log_dir: str = "logs",
        log_file_pattern: str = "system_log_{date}.log",
    ):
        self.service_name = service_name
        self.server_ip = get_server_ip()
        self.log_dir = log_dir
        self.log_file_pattern = log_file_pattern

        os.makedirs(self.log_dir, exist_ok=True)

        self.console_logger = logging.getLogger("bpjs_standard_logger_console")
        self.console_logger.setLevel(logging.INFO)
        self.console_logger.propagate = False

        if not self.console_logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self.console_logger.addHandler(console_handler)

    def get_current_log_path(self) -> str:
        """Resolve current day log file path using ddmmyyyy format."""
        date_str = get_wib_date_str()
        filename = self.log_file_pattern.replace("{date}", date_str)
        return os.path.join(self.log_dir, filename)

    def write_system_log(
        self,
        log_level: str,
        transaction_id: str,
        trace_id: str,
        endpoint: str,
        method_type: str,
        content_type: str,
        function_name: str,
        caller_info: str,
        execution_time: float,
        client_ip: str,
        event_name: str,
        activity: str,
        obj_name: str,
        res_desc: str,
        short_desc: str,
        flag_start_stop: str,
        extra_detail: str = "",
        error: str = "",
        protocol: str = "REST",
        execution_type: str = "SYNC",
        prev_transaction_id: str = "",
        body: Any = None,
        result: Any = None,
    ) -> str:
        """Construct and write a standard single-line log without JSON braces or escaped quotes."""

        timestamp = get_wib_timestamp()
        flag_tag = "| START |" if "START" in flag_start_stop.upper() else "| STOP |"
        error_part = f"'{error}'" if error else "''"
        message_part = f"'{activity}' on '{obj_name}' with result '{res_desc}' with error {error_part}. {short_desc}"

        extra_str = f" {extra_detail}" if extra_detail else ""

        # Normalize TransactionID and TraceID format
        tr_id = transaction_id if transaction_id.startswith("TR-") else f"TR-{transaction_id.replace('tr-', '')}"
        tc_id = trace_id if trace_id.startswith("TC-") else f"TC-{trace_id.replace('tc-', '')}"

        # Clean string format without {} and escaped double quotes
        log_line = (
            f"{timestamp} [{log_level.upper()}] {tr_id} {self.service_name} "
            f"{endpoint} {protocol} {method_type} {execution_type} {content_type or 'application/json'} "
            f"{function_name} {caller_info} {execution_time:.2f} ms {self.server_ip} {client_ip} "
            f"{event_name} {tc_id}{extra_str} {flag_tag} {message_part}"
        )

        # Write to daily ddmmyyyy log file
        log_path = self.get_current_log_path()
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write system log to %s", log_path)

        # Console output
        self.console_logger.info(log_line)

        return log_line
