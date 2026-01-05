import streamlit as st
import sqlite3
import hashlib
import uuid
from datetime import datetime, date, timedelta, timezone
import pandas as pd
from typing import Optional, Dict, List
import html
import asyncio
import threading
import json
import websockets  # pip install websockets
import os
import time
import stripe
import io
import urllib.parse
import qrcode
from PIL import Image
from fpdf import FPDF
import calendar
import os
import sys          # ✅ REQUIRED for EXE detection
import sqlite3
import webbrowser
import time
import os

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"



def get_app_data_dir():
    """
    Returns a writable directory for DB storage.
    Works for both .py and PyInstaller .exe
    """
    if getattr(sys, 'frozen', False):
        # Running as EXE
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        app_dir = os.path.join(base, "QuePOS")
    else:
        # Running as normal Python script
        app_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(app_dir, exist_ok=True)
    return app_dir


# ✅ SAFE DB PATH (SINGLE SOURCE OF TRUTH)
DB_PATH = os.path.join(get_app_data_dir(), "quepos.db")


# ------------- STRIPE CONFIG -------------
# Set these in .streamlit/secrets.toml:
# STRIPE_API_KEY = "sk_live_..."
# STRIPE_PUBLISHABLE_KEY = "pk_live_..."
# STRIPE_SUCCESS_URL = "http://localhost:8501/?checkout=success"
# STRIPE_CANCEL_URL = "http://localhost:8501/?checkout=cancel"

STRIPE_API_KEY = st.secrets.get("STRIPE_API_KEY", "")
STRIPE_PUBLISHABLE_KEY = st.secrets.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_SUCCESS_URL = st.secrets.get("STRIPE_SUCCESS_URL", "http://localhost:8501/?checkout=success")
STRIPE_CANCEL_URL = st.secrets.get("STRIPE_CANCEL_URL", "http://localhost:8501/?checkout=cancel")

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY
else:
    # App still works for cash/card if Stripe not configured
    stripe = None

def create_stripe_checkout_session(order_id: str, amount: float, currency: str = "gbp") -> Optional[str]:
    """
    Create a Stripe Checkout Session for an order and return the hosted payment URL.
    If Stripe is not configured, returns None and shows an error.
    """
    if not STRIPE_API_KEY or stripe is None:
        st.error("Stripe is not configured. Set STRIPE_API_KEY and STRIPE_PUBLISHABLE_KEY in .streamlit/secrets.toml.")
        return None

    try:
        # Stripe expects the amount in the smallest currency unit (pence for GBP)
        amount_int = int(round(amount * 100))

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="payment",
            line_items=[
                {
                    "price_data": {
                        "currency": currency,
                        "product_data": {
                            "name": f"Order {order_id[:8]}",
                        },
                        "unit_amount": amount_int,
                    },
                    "quantity": 1,
                }
            ],
            success_url=f"{STRIPE_SUCCESS_URL}&order_id={order_id}",
            cancel_url=f"{STRIPE_CANCEL_URL}&order_id={order_id}",
            metadata={"order_id": order_id},
        )

        return checkout_session.url

    except Exception as e:
        st.error(f"Stripe error: {e}")
        return None


# -------------------------------------------------------------------
# BASIC CONFIG
# -------------------------------------------------------------------

LOCK_FILE = os.path.join(get_app_data_dir(), "license.lock")
WS_URL = "ws://localhost:8765"

# -------------------------------------------------------------------
# LICENSE LOCK HELPERS
# -------------------------------------------------------------------

def save_license_lock(data: dict):
    """Store license info outside DB to prevent bypass by deleting DB."""
    with open(LOCK_FILE, "w") as f:
        json.dump(data, f)

def load_license_lock():
    if not os.path.exists(LOCK_FILE):
        return None
    try:
        with open(LOCK_FILE, "r") as f:
            return json.load(f)
    except:
        return None

# -------------------------------------------------------------------
# DB HELPERS
# -------------------------------------------------------------------


def generate_qr_from_link(qr_link: str, name: str):
    """
    Generate and save QR code image as PNG.
    - Works in Python + PyInstaller EXE
    - Uses app writable directory
    """
    base_dir = get_app_data_dir()

    qr_dir = os.path.join(base_dir, "qr_codes")
    os.makedirs(qr_dir, exist_ok=True)

    img = qrcode.make(qr_link)   # PNG by default
    path = os.path.join(qr_dir, f"{name}.png")
    img.save(path)

    return path

def get_conn():
    """
    Single place to open SQLite connection.

    - Uses WAL journal mode for better concurrency
    - Enables foreign keys
    - Uses Row factory so existing fetchone()/fetchall() code keeps working
    """
    conn = sqlite3.connect(
        DB_PATH,                 # ✅ SINGLE SOURCE OF TRUTH
        timeout=30,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row

    # Core pragmas – safe to run every time
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    return conn

def execute(query: str, params: tuple = ()):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("BEGIN")
        cur.execute(query, params)
        conn.commit()
        return cur
    except sqlite3.DatabaseError as e:
        st.error(f"DB Error: {e}")
        conn.rollback()
        return None

def fetchall(query: str, params: tuple = ()):
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(query, params)
        rows = c.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetchone(query: str, params: tuple = ()):
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(query, params)
        row = c.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_active_table_order(table_no: str):
    """
    SINGLE active order for a table (current seating).

    - Only counts orders that are:
        • NOT closed / rejected
        • NOT paid
        • And either:
            - DINE with at least one item
            - QR that is already approved
    """
    return fetchone(
        """
        SELECT o.*
        FROM orders o
        WHERE o.table_no = ?
          AND o.status NOT IN ('closed','rejected')
          AND COALESCE(o.paid, 0) = 0
          AND (
                (
                    o.order_type = 'dine'
                    AND EXISTS (
                        SELECT 1 FROM order_items oi
                        WHERE oi.order_id = o.id
                    )
                )
             OR (
                    o.order_type = 'qr'
                    AND o.approved = 1
                )
          )
        ORDER BY o.created_at DESC
        LIMIT 1
        """,
        (table_no,),
    )


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Simple SHA-256 check. Compatible with your existing hash_password().
    """
    return hash_password(plain_password) == stored_hash


# ============================================================
# 🔐 QR SECURITY HELPERS — HMAC SIGNATURE (STATIC + LENIENT)
# ============================================================

import hmac, hashlib

def generate_qr_sig(table_no: str) -> str:
    """
    Generate a stable HMAC signature for a given table number.
    Used to protect QR links from tampering.
    """
    secret = st.secrets.get("HMAC_SECRET_KEY", "default_secret")
    return hmac.new(secret.encode(), table_no.encode(), hashlib.sha256).hexdigest()[:12]


def validate_qr_sig(table_no: str, sig: str, strict: bool = False) -> bool:
    """
    Validate HMAC signature for QR link.

    Supports:
    - Legacy 12-char signatures
    - Current HMAC-based signatures

    strict=False → allow access but warn
    strict=True  → block on mismatch
    """
    if not table_no:
        return False

    # No signature provided
    if not sig:
        if strict:
            return False
        else:
            print(f"[QR SECURITY] No signature for table {table_no}, leniently allowed.")
            return True

    # Secret (safe fallback for local / cloud)
    secret = st.secrets.get("HMAC_SECRET_KEY", "default_secret")

    # Generate expected signatures (support legacy + new)
    expected_full = hmac.new(
        secret.encode(),
        table_no.encode(),
        hashlib.sha256
    ).hexdigest()

    expected_12 = expected_full[:12]
    expected_16 = expected_full[:16]

    # Accept any known valid format
    if sig in (expected_12, expected_16, expected_full):
        return True

    # Mismatch handling
    if strict:
        print(f"[QR SECURITY] Signature mismatch for table {table_no} — strict reject.")
        return False
    else:
        print(f"[QR SECURITY] Signature mismatch for table {table_no} — leniently allowed.")
        return True





# -------------------------------------------------------------------
# REALTIME (WEBSOCKETS)
# -------------------------------------------------------------------

@st.cache_resource
def start_ws_listener():
    """Start a background thread for this Streamlit session."""
    def _run():
        asyncio.run(ws_client_loop())
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True

async def ws_client_loop():
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                await ws.send(json.dumps({"type": "hello", "who": "pos_client"}))
                async for message in ws:
                    try:
                        data = json.loads(message)
                    except:
                        data = {"raw": message}
                    st.session_state["__last_rt_event__"] = data
                    st.rerun()
        except Exception:
            await asyncio.sleep(2)

def push_realtime_event(event_type: str, payload: dict):
    async def _send():
        try:
            async with websockets.connect(WS_URL) as ws:
                msg = {"type": event_type, "payload": payload}
                await ws.send(json.dumps(msg))
        except Exception:
            pass

    threading.Thread(target=lambda: asyncio.run(_send()), daemon=True).start()

# -------------------------------------------------------------------
# LICENSE HELPERS
# -------------------------------------------------------------------

import re
import hashlib

def generate_license_key(customer_name: str, expiry_year: int, expiry_month: int) -> str:
    """
    Stable, production-safe license generator.
    Must match your external generator script exactly.
    """
    # Normalise customer name: collapse spaces, trim, uppercase
    clean_name = re.sub(r"\s+", " ", customer_name).strip().upper()
    base = f"{clean_name}|{expiry_year:04d}{expiry_month:02d}|QPOS_V1"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16].upper()


def get_active_license():
    row = fetchone("SELECT * FROM license ORDER BY id DESC LIMIT 1")
    if not row:
        return None
    exp_str = row.get("expires_on")
    is_expired = False
    if exp_str:
        try:
            exp_date = datetime.fromisoformat(exp_str).date()
            if exp_date < date.today():
                is_expired = True
        except Exception:
            pass
    row["is_expired"] = is_expired
    return row

def check_license_and_date():
    """
    Extra protection: if system date goes backwards compared to last_run_date,
    we treat license as invalid.
    """
    lic = get_active_license()
    if not lic:
        return False, "No license found. Please activate your license."

    if lic.get("is_expired"):
        return False, f"License expired on {lic.get('expires_on', '')}. Please renew."

    last_run = lic.get("last_run_date")
    today = date.today()

    if last_run:
        try:
            last_run_date = datetime.fromisoformat(last_run).date()
            if today < last_run_date:
                execute(
                    "UPDATE license SET expires_on=?, last_run_date=? WHERE id=?",
                    (today.isoformat(), today.isoformat(), lic["id"]),
                )
                return False, (
                    "System date moved backwards.\n\n"
                    "License locked. Please contact support for a new key."
                )
        except Exception:
            pass

    execute(
        "UPDATE license SET last_run_date=? WHERE id=?",
        (today.isoformat(), lic["id"]),
    )
    return True, ""

def show_expiry_warning_if_needed(active_license):
    if not active_license:
        return
    exp_str = active_license.get("expires_on")
    if not exp_str:
        return
    try:
        exp_date = datetime.fromisoformat(exp_str).date()
    except Exception:
        return

    today = date.today()
    days_left = (exp_date - today).days
    if 0 < days_left <= 7:
        if st.session_state.get("expiry_warn_shown") == today.isoformat():
            return
        st.warning(
            f"⚠️ Your license will expire in **{days_left} day(s)** "
            f"on **{exp_date.strftime('%Y-%m-%d')}**. Please renew soon."
        )
        st.session_state["expiry_warn_shown"] = today.isoformat()
def license_activation_screen():
    """
    License activation UI.

    • Asks for Customer Name, Expiry (Year+Month) and License Key
    • Validates using generate_license_key()
    • Stores SINGLE license in DB
    • Forces rerun so app proceeds correctly
    """

    st.markdown(
        "<h2 style='text-align:center;margin-bottom:0.5rem;'>Que-POS License Activation</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;margin-bottom:1.5rem;'>"
        "Enter your license details exactly as given on your license invoice."
        "</p>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # EXISTING VALID LICENSE
    # ------------------------------------------------------------
    existing = get_active_license()
    if existing and not existing.get("is_expired", False):
        st.success(
            f"Existing license: **{existing['customer_name']}**, "
            f"valid till **{existing['expires_on']}**."
        )
        if st.button("Continue to Login"):
            st.rerun()   # ✅ REQUIRED
        return

    # ------------------------------------------------------------
    # PREFILL FROM LOCK FILE
    # ------------------------------------------------------------
    lock = load_license_lock() or {}
    default_customer = lock.get("customer_name", "")
    default_key = lock.get("license_key", "")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        customer_name = st.text_input("Customer Name", value=default_customer)
    with col2:
        expiry_year = st.number_input(
            "Expiry Year (YYYY)",
            min_value=2024,
            max_value=2100,
            value=datetime.today().year,
            step=1,
        )
    with col3:
        expiry_month = st.number_input(
            "Expiry Month (1-12)",
            min_value=1,
            max_value=12,
            value=datetime.today().month,
            step=1,
        )

    license_key = st.text_input(
        "License Key",
        value=default_key,
        help="16-character code (no spaces).",
    )

    st.markdown(
        "<p style='font-size:0.9rem;color:#9ca3af;'>"
        "The license key is generated using Customer Name + Expiry Month/Year. "
        "Both must match exactly."
        "</p>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # ACTIVATE
    # ------------------------------------------------------------
    if st.button("🔓 Activate License", width="stretch"):
        name_clean = customer_name.strip()
        key_clean = license_key.strip().upper()

        if not name_clean or not key_clean:
            st.error("Please fill in all fields.")
            return

        try:
            exp_year = int(expiry_year)
            exp_month = int(expiry_month)
            if not (1 <= exp_month <= 12):
                raise ValueError("Invalid month")

            expected_key = generate_license_key(name_clean, exp_year, exp_month)
            if key_clean != expected_key:
                st.error("❌ Invalid license key for this customer and expiry.")
                return

            # Expiry = last day of month
            last_day = calendar.monthrange(exp_year, exp_month)[1]
            expires_on = date(exp_year, exp_month, last_day)

            # ----------------------------------------------------
            # 🔥 ENSURE SINGLE LICENSE ROW
            # ----------------------------------------------------
            execute("DELETE FROM license")

            execute(
                """
                INSERT INTO license
                (license_key, customer_name, expires_on, created_at, last_run_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key_clean,
                    name_clean,
                    expires_on.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    date.today().isoformat(),
                ),
            )

            save_license_lock(
                {
                    "license_key": key_clean,
                    "customer_name": name_clean,
                    "expires_on": expires_on.isoformat(),
                }
            )

            st.success(
                f"✅ License activated for **{name_clean}** until **{expires_on.isoformat()}**."
            )

            time.sleep(0.5)
            st.rerun()   # ✅ REQUIRED

        except Exception as e:
            st.error(f"Activation failed: {e}")


def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(password: str, stored_hash: str) -> bool:
    """Validate password by comparing SHA256 hashes."""
    return hash_password(password) == stored_hash

# -------------------------------------------------------------------
# AUTH
# -------------------------------------------------------------------

def login(username, password):
    import json

    user = fetchone("SELECT * FROM users WHERE username=?", (username,))
    if not user:
        return False

    # password check using our verify helper
    if not verify_password(password, user["password_hash"]):
        return False

    allowed_modules = []
    raw_modules = user.get("allowed_modules")
    if raw_modules:
        try:
            allowed_modules = json.loads(raw_modules)
        except Exception:
            allowed_modules = []

    st.session_state["user"] = {
        "id": user["id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user["role"],
        "allowed_modules": allowed_modules,
    }
    return True


def logout():
    for k in [
        "user",
        "open_table_order_id",
        "open_takeaway_order_id",
        "_last_table_selected",
        "_last_table_selected_touch",
        "_inv_editing",
        "show_add_menu",
        "sidebar",
        "go_home",
        "nav_override",
        "home_target",
    ]:
        if k in st.session_state:
            del st.session_state[k]

# -------------------------------------------------------------------
# DB INIT / MIGRATIONS
# -------------------------------------------------------------------

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # ============================================================
    # SQLITE SAFETY / PERFORMANCE PRAGMAS
    # ============================================================
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA busy_timeout=5000;")

    # ============================================================
    # USERS
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            full_name TEXT,
            role TEXT,
            allowed_modules TEXT
        )
    """)

    # ============================================================
    # MENU
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS menu (
            id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            tax REAL DEFAULT 0.0,
            stock INTEGER DEFAULT 0,
            available INTEGER DEFAULT 1,
            notes TEXT
        )
    """)

    # ============================================================
    # ORDERS
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            order_type TEXT,
            table_no TEXT,
            customer_name TEXT,
            token TEXT,
            status TEXT DEFAULT 'pending',
            subtotal REAL DEFAULT 0,
            tax REAL DEFAULT 0,
            total REAL DEFAULT 0,
            paid INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            currency TEXT DEFAULT 'GBP',
            approved INTEGER DEFAULT 0
        )
    """)

    # ============================================================
    # ORDER ITEMS
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id TEXT PRIMARY KEY,
            order_id TEXT,
            menu_id TEXT,
            name TEXT,
            qty INTEGER,
            unit_price REAL,
            tax REAL,
            note TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(order_id) REFERENCES orders(id) ON DELETE CASCADE
        )
    """)

    # ============================================================
    # PAYMENTS
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id TEXT PRIMARY KEY,
            order_id TEXT,
            method TEXT,
            amount REAL,
            tax_amount REAL DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY(order_id) REFERENCES orders(id) ON DELETE CASCADE
        )
    """)

    c.execute("PRAGMA table_info(payments)")
    if "tax_amount" not in [r[1] for r in c.fetchall()]:
        c.execute("ALTER TABLE payments ADD COLUMN tax_amount REAL DEFAULT 0")

    # ============================================================
    # KDS
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS kds_items_status (
            id TEXT PRIMARY KEY,
            order_item_id TEXT,
            status TEXT DEFAULT 'pending',
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(order_item_id) REFERENCES order_items(id) ON DELETE CASCADE
        )
    """)

    # ============================================================
    # LICENSE (FULLY MIGRATION-SAFE)
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS license (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_key TEXT,
            customer_name TEXT,
            expires_on TEXT,
            created_at TEXT,
            last_run_date TEXT,
            is_expired INTEGER DEFAULT 0,
            machine_id TEXT
        )
    """)

    c.execute("PRAGMA table_info(license)")
    lic_cols = [r[1] for r in c.fetchall()]

    if "created_at" not in lic_cols:
        c.execute("ALTER TABLE license ADD COLUMN created_at TEXT")

    if "last_run_date" not in lic_cols:
        c.execute("ALTER TABLE license ADD COLUMN last_run_date TEXT")

    if "is_expired" not in lic_cols:
        c.execute("ALTER TABLE license ADD COLUMN is_expired INTEGER DEFAULT 0")

    if "machine_id" not in lic_cols:
        c.execute("ALTER TABLE license ADD COLUMN machine_id TEXT")

    # ============================================================
    # BUSINESS PROFILE
    # ============================================================
    c.execute("""
        CREATE TABLE IF NOT EXISTS business_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            business_name TEXT,
            address TEXT,
            phone TEXT,
            vat_no TEXT,
            closing_line TEXT,
            currency TEXT DEFAULT 'GBP',
            currency_locked INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        INSERT OR IGNORE INTO business_profile
        (id, business_name, currency, currency_locked)
        VALUES (1, 'My Business', 'GBP', 0)
    """)

    # ============================================================
    # DEFAULT SUPERADMIN
    # ============================================================
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO users
            (id, username, password_hash, full_name, role, allowed_modules)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            "admin",
            hash_password("admin123"),
            "Default Admin",
            "superadmin",
            json.dumps([
                "Home", "Orders", "KDS", "Inventory",
                "Menu", "Payments", "Reports", "Admin", "Takeaway Status"
            ])
        ))

    conn.commit()
    conn.close()



# -------------------------------------------------------------------
# ORDER HELPERS
# -------------------------------------------------------------------

def next_token() -> str:
    row = fetchone("SELECT token FROM orders WHERE token IS NOT NULL ORDER BY created_at DESC LIMIT 1")
    if not row or not row.get("token"):
        return "A0001"
    last = row["token"]
    if len(last) != 5 or not last[0].isalpha() or not last[1:].isdigit():
        return "A0001"
    letter = last[0].upper()
    num = int(last[1:])
    num += 1
    if num > 9999:
        letter = "A" if letter == "Z" else chr(ord(letter) + 1)
        num = 1
    return f"{letter}{num:04d}"

def create_user(
    username: str,
    password: str,
    full_name: str = "",
    role: str = "cashier",
    allowed_modules: list | None = None,
):
    import json

    # Default module access (non-superadmin)
    default_modules = ["Home", "Orders", "KDS", "Takeaway Status", "Inventory"]

    # Full access list
    full_modules = [
        "Home",
        "Orders",
        "KDS",
        "Takeaway Status",
        "Inventory",
        "Menu",
        "Payments",
        "Reports",
        "Admin",
    ]

    # Decide what to store
    if role == "superadmin":
        modules_to_store = full_modules
    else:
        modules_to_store = allowed_modules if allowed_modules else default_modules

    try:
        execute(
            """
            INSERT INTO users (id, username, password_hash, full_name, role, allowed_modules)
            VALUES (?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                username.strip(),
                hash_password(password),
                full_name.strip(),
                role,
                json.dumps(modules_to_store),
            ),
        )
        st.success(f"User '{username}' created successfully.")
    except sqlite3.IntegrityError:
        st.warning("Username already exists.")

# ✅ SIMPLE PERMISSION CHECK (used by all panels)
def user_can_access(module_name: str) -> bool:
    """Check if current user has permission to access a given module."""
    import json
    user = st.session_state.get("user", {})
    allowed = user.get("allowed_modules", [])
    if isinstance(allowed, str):
        try:
            allowed = json.loads(allowed)
        except Exception:
            allowed = []
    return module_name in allowed


# ============================================================
# 🔐 USER ACCESS CONTROL HELPERS (NEW)
# ============================================================

# Define what each module can do
MODULE_ACTIONS = {
    "Home": ["view"],
    "Orders": ["view", "create", "bill"],
    "KDS": ["view", "update"],
    "Takeaway Status": ["view"],
    "Inventory": ["view", "update"],
    "Menu": ["view", "update"],
    "Payments": ["view", "bill"],
    "Reports": ["view"],
    "Admin": ["view", "update", "approve"],
}

def get_current_user():
    """Return the logged-in user dict from session_state, or None."""
    return st.session_state.get("user")

def get_current_role():
    """Return current user's role name."""
    u = get_current_user()
    return u.get("role") if u else None

def get_allowed_modules() -> list:
    """Return allowed modules list for current user."""
    u = get_current_user()
    if not u:
        return []
    try:
        return json.loads(u.get("allowed_modules", "[]"))
    except Exception:
        return []

def can(action: str) -> bool:
    """
    Check whether current user is allowed to perform an action.
    e.g. if can("bill"):  # then allow billing access
    """
    modules = get_allowed_modules()
    for m in modules:
        if action in MODULE_ACTIONS.get(m, []):
            return True
    return False

def allowed_modules_for_user(user=None) -> list:
    """
    Helper to safely parse allowed_modules for any user dict.
    """
    if not user:
        user = get_current_user()
    try:
        return json.loads(user.get("allowed_modules", "[]"))
    except Exception:
        return []

def create_order(order_type: str, table_no: Optional[str], customer_name: Optional[str], paid: bool = False) -> str:
    order_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    token = None
    status = "pending"
    paid_flag = 0

    # 🔒 DEFAULT: NOT approved
    approved_flag = 0

    if order_type == "takeaway":
        token = next_token()
        approved_flag = 1
        if paid:
            status = "paid"
            paid_flag = 1

    elif order_type == "dine":
        approved_flag = 1

    elif order_type == "qr":
        approved_flag = 0
        status = "pending"

    execute(
        """
        INSERT INTO orders
        (id, order_type, table_no, customer_name, token, created_at, status, total, paid, approved)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            order_id,
            order_type,
            table_no,
            customer_name,
            token,
            now,
            status,
            0.0,
            paid_flag,
            approved_flag,
        ),
    )

    return order_id

def get_or_create_qr_order(table_no: str) -> str:
    """
    Reuse ONLY existing OPEN QR order for a table.
    Never attach directly to DINE-IN (approval gate).
    """
    row = fetchone(
        """
        SELECT id FROM orders
        WHERE table_no = ?
          AND order_type = 'qr'
          AND approved = 0
          AND status NOT IN ('closed','paid','rejected')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (table_no,),
    )

    if row:
        return row["id"]

    # Create NEW QR order (needs approval)
    order_id = str(uuid.uuid4())
    execute(
        """
        INSERT INTO orders
        (id, order_type, table_no, status, approved, created_at, updated_at)
        VALUES (?, 'qr', ?, 'pending', 0, datetime('now'), datetime('now'))
        """,
        (order_id, table_no),
    )

    return order_id


def ensure_dine_order_for_table(table_no: str) -> str:
    """
    Find an existing active dine-in order for this table, or create a new one.
    Returns the order_id.
    """
    table_no = str(table_no).upper()

    existing = fetchone(
        """
        SELECT * FROM orders
        WHERE order_type='dine'
          AND table_no=?
          AND status IN ('pending','in_progress','ready','served')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (table_no,),
    )

    if existing:
        return existing["id"]

    # No active order → create fresh dine-in order for that table
    return create_order(order_type="dine", table_no=table_no, customer_name=None, paid=False)


def cleanup_rejected_orders():
    """
    Permanently delete rejected orders (approved = -1) and all related data.
    Safe cleanup logic — removes from:
      • orders
      • order_items
      • kds_items_status
      • payments
    Run this periodically (e.g. every 10–30 minutes) via main() or admin tools.
    """
    try:
        # Find rejected order IDs
        rejected_orders = fetchall(
            "SELECT id FROM orders WHERE approved=-1 OR status='cancelled'"
        )
        if not rejected_orders:
            return 0

        count = 0
        for o in rejected_orders:
            oid = o["id"]

            # Delete related records manually (to ensure no orphan data)
            execute("DELETE FROM payments WHERE order_id=?", (oid,))
            execute(
                """
                DELETE FROM kds_items_status
                WHERE order_item_id IN (
                    SELECT id FROM order_items WHERE order_id=?
                )
                """,
                (oid,),
            )
            execute("DELETE FROM order_items WHERE order_id=?", (oid,))
            execute("DELETE FROM orders WHERE id=?", (oid,))

            count += 1

        if count > 0:
            print(f"🧹 Cleaned up {count} rejected order(s).")

        return count

    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")
        return 0

# -------------------------------------------------------------------
# 🔁 REALTIME EVENT SYSTEM (file-based, auto-clean)
# -------------------------------------------------------------------
import json, time, os, streamlit as st

REALTIME_EVENT_FILE = os.path.join(get_app_data_dir(), "realtime_event.json")
EVENT_TTL_SECONDS = 60  # auto-delete event file if older than 60s

def push_realtime_event(event_type: str, payload: dict):
    """
    Write a small JSON 'event file' that get_realtime_event() will poll.
    No websockets, no asyncio – works fine in EXE.
    """
    try:
        # Ensure folder exists (in case REALTIME_EVENT_FILE is in a subdir)
        event_dir = os.path.dirname(REALTIME_EVENT_FILE)
        if event_dir:
            os.makedirs(event_dir, exist_ok=True)
    except Exception:
        # Folder creation failure is non-fatal here
        pass

    try:
        data = {
            "type": event_type,
            "data": payload,
            "timestamp": time.time(),
        }
        with open(REALTIME_EVENT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        # Optional debug
        # print(f"[push_realtime_event] wrote event: {data}")
    except Exception as e:
        print(f"[push_realtime_event] Error: {e}")

def get_realtime_event():
    """
    Reads the latest event written by push_realtime_event().
    Returns a dict like {"type": "bill_request", "data": {...}} or None.
    Auto-cleans the event file if expired (> EVENT_TTL_SECONDS).
    """
    try:
        if not os.path.exists(REALTIME_EVENT_FILE):
            return None

        # Auto-delete old file
        age = time.time() - os.path.getmtime(REALTIME_EVENT_FILE)
        if age > EVENT_TTL_SECONDS:
            os.remove(REALTIME_EVENT_FILE)
            print(f"[Realtime Event] Cleared old event file ({int(age)}s old)")
            return None

        with open(REALTIME_EVENT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        last_seen = st.session_state.get("_last_event_seen")
        if data and data.get("timestamp") != last_seen:
            st.session_state["_last_event_seen"] = data.get("timestamp")
            return data
        return None

    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[get_realtime_event] Error: {e}")
        return None


def add_order_item(order_id: str, menu_row: dict, qty: int, note: str = "") -> bool:
    order = fetchone("SELECT * FROM orders WHERE id=?", (order_id,))
    if not order:
        st.error("Order not found")
        return False

    # ❌ never allow items on closed / paid orders
    if order["status"] in ("closed", "paid"):
        st.warning("Cannot add items to a closed or paid order")
        return False

    latest_menu = fetchone("SELECT * FROM menu WHERE id=?", (menu_row["id"],))
    if not latest_menu:
        st.error("Menu item no longer exists")
        return False

    if int(latest_menu.get("stock") or 0) <= 0:
        st.warning(f"{latest_menu['name']} is out of stock!")
        return False

    # ------------------------------------------------------------
    # INSERT ORDER ITEM
    # ------------------------------------------------------------
    execute(
        """
        INSERT INTO order_items
        (id, order_id, menu_id, name, qty, unit_price, tax, note)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            str(uuid.uuid4()),
            order_id,
            latest_menu["id"],
            latest_menu["name"],
            qty,
            latest_menu["price"],
            latest_menu["tax"],
            note,
        ),
    )

    # ------------------------------------------------------------
    # STOCK REDUCTION (UNCHANGED)
    # ------------------------------------------------------------
    new_stock = max(0, int(latest_menu["stock"]) - int(qty))
    execute("UPDATE menu SET stock=? WHERE id=?", (new_stock, latest_menu["id"]))

    # ------------------------------------------------------------
    # 🔥 UNIVERSAL KDS ENTRY RULE (STANDARDISED)
    # ------------------------------------------------------------
    if order["status"] == "pending" and order["order_type"] in ("dine", "takeaway", "qr"):
        execute(
            "UPDATE orders SET status='in_progress', updated_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), order_id),
        )
        push_realtime_event("order_to_kds", {"order_id": order_id})

    # ------------------------------------------------------------
    # TOTAL RECALCULATION ONLY
    # ------------------------------------------------------------
    line_total = qty * latest_menu["price"] * (1 + latest_menu["tax"] / 100.0)
    total_row = fetchone("SELECT total FROM orders WHERE id=?", (order_id,))
    total = float(total_row["total"] or 0.0) if total_row else 0.0
    total += line_total

    execute(
        "UPDATE orders SET total=?, updated_at=? WHERE id=?",
        (total, datetime.now(timezone.utc).isoformat(), order_id),
    )

    push_realtime_event("order_updated", {"order_id": order_id})
    return True

# -------------------------------------------------------------------
# TABLE / TAKEAWAY ORDER DETAIL SCREENS
# -------------------------------------------------------------------

def show_table_order(order_id: str, menu_items):
    order = fetchone("SELECT * FROM orders WHERE id=?", (order_id,))
    if not order:
        st.error("Order not found")
        return

    st.markdown(
        f"<div class='section-card'><h3>Table Order • {order.get('table_no') or ''} "
        f"• <small class='small-muted'>Status: {order['status'].upper()}</small></h3>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # 🔒 STATUS HANDLING (NO QR→DINE CONVERSION HERE)
    # ============================================================

    if order["status"] == "bill_requested":
        st.warning("💰 Bill has been requested by the customer. Please review and close the table when ready.")

    elif order["status"] in ("closed", "paid"):
        st.info("Order closed / paid. Bill below.")
        show_bill(order_id)
        st.session_state.pop("open_table_order_id", None)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    elif order["status"] == "ready":
        st.info("Order is ready (all items completed in KDS). You can now raise the bill.")

    elif order["status"] == "served":
        st.info("Order has been served. You can still add items or close & print bill.")



    # ============================================================
    # 💾 SHOW EXISTING ITEMS
    # ============================================================
    items = fetchall("SELECT * FROM order_items WHERE order_id=?", (order_id,))
    if items:
        df = pd.DataFrame(items)
        df["tax_amount"] = df["qty"] * df["unit_price"] * (df["tax"] / 100)
        df["line_total"] = df["qty"] * df["unit_price"] + df["tax_amount"]

        kds_statuses = {}
        for item in items:
            status_row = fetchone(
                "SELECT status FROM kds_items_status WHERE order_item_id=?",
                (item["id"],),
            )
            kds_statuses[item["id"]] = status_row["status"] if status_row else "pending"
        df["kds_status"] = df["id"].map(kds_statuses)

        st.dataframe(
            df[
                [
                    "name",
                    "qty",
                    "unit_price",
                    "tax",
                    "tax_amount",
                    "line_total",
                    "note",
                    "kds_status",
                ]
            ].rename(
                columns={
                    "tax": "Tax %",
                    "tax_amount": "Tax Amt",
                    "line_total": "Line Total",
                    "kds_status": "KDS Status",
                }
            ),
            width="stretch",
        )

    # ============================================================
    # 💷 TOTAL + ACTION BUTTONS
    # ============================================================
    total_row = fetchone("SELECT total FROM orders WHERE id=?", (order_id,))
    total = total_row["total"] or 0.0 if total_row else 0.0
    st.markdown(f"**Order total (incl. tax): £{total:.2f}**")

    col1, col2, col3 = st.columns([1, 1, 2])

    if col1.button("Mark In Progress", key=f"tbl_prog_{order_id}"):
        execute("UPDATE orders SET status=? WHERE id=?", ("in_progress", order_id))
        push_realtime_event("order_updated", {"order_id": order_id})
        st.rerun()

    if col2.button("Close Table & Print Bill", key=f"tbl_close_{order_id}"):
        execute(
            "UPDATE orders SET status=?, updated_at=? WHERE id=?",
            ("closed", datetime.now(timezone.utc).isoformat(), order_id),
        )
        push_realtime_event("order_closed", {"order_id": order_id})
        st.session_state["__bill_to_show__"] = order_id
        st.session_state.pop("open_table_order_id", None)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def show_takeaway_order(order_id: str, menu_items):
    order = fetchone("SELECT * FROM orders WHERE id=?", (order_id,))
    if not order:
        st.error("Order not found")
        return

    st.markdown(
        f"<div class='section-card'><h3>Takeaway Order • Token: {order.get('token') or ''} "
        f"• <small class='small-muted'>Status: {order['status'].upper()}</small></h3>",
        unsafe_allow_html=True,
    )

    # -------- CLOSED or PAID (unchanged) --------
    if order["status"] in ("closed", "paid"):
        st.info("Order closed / paid. Bill below.")
        show_bill(order_id)
        st.session_state.pop("open_takeaway_order_id", None)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if order["status"] == "served":
        st.info("Order has been served (KDS). You can still add items or close & print bill.")



    # -------- ITEMS LIST --------
    items = fetchall("SELECT * FROM order_items WHERE order_id=?", (order_id,))
    if items:
        df = pd.DataFrame(items)
        df["tax_amount"] = df["qty"] * df["unit_price"] * (df["tax"] / 100)
        df["line_total"] = df["qty"] * df["unit_price"] + df["tax_amount"]

        st.dataframe(
            df[
                ["name", "qty", "unit_price", "tax", "tax_amount", "line_total", "note"]
            ].rename(
                columns={
                    "tax": "Tax %",
                    "tax_amount": "Tax Amt",
                    "line_total": "Line Total",
                }
            ),
            width="stretch",
        )

    # -------- TOTAL --------
    total_row = fetchone("SELECT total FROM orders WHERE id=?", (order_id,))
    total = total_row["total"] or 0.0 if total_row else 0.0
    st.markdown(f"**Order total (incl. tax): £{total:.2f}**")

    # -------- BUTTONS --------
    col1, col2 = st.columns(2)

    # Mark In Progress
    if col1.button("Mark In Progress", key=f"tky_prog_{order_id}"):
        execute("UPDATE orders SET status=?, updated_at=? WHERE id=?",
                ("in_progress", datetime.now(timezone.utc).isoformat(), order_id))
        st.rerun()

    # -------- CLOSE & PRINT BILL (FIXED) --------
    if col2.button("Close & Print Bill", key=f"tky_close_{order_id}"):

        # FIX: Takeaway should NOT close before payment
        new_status = "ready"   # instead of "closed"

        execute(
            "UPDATE orders SET status=?, updated_at=? WHERE id=?",
            (new_status, datetime.now(timezone.utc).isoformat(), order_id),
        )

        # Show bill preview
        st.session_state["__bill_to_show__"] = order_id

        # Remove from active session
        st.session_state.pop("open_takeaway_order_id", None)

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# RECEIPT HELPERS
# -------------------------------------------------------------------

def get_setting(key: str, default=None):
    return default

def get_order_with_items(order_id: str):
    order = fetchone("SELECT * FROM orders WHERE id=?", (order_id,))
    if not order:
        return None
    items = fetchall("SELECT * FROM order_items WHERE order_id=?", (order_id,))
    subtotal = 0.0
    tax_total = 0.0
    for it in items:
        base = it["qty"] * it["unit_price"]
        line_tax = base * (it["tax"] / 100)
        subtotal += base
        tax_total += line_tax
    total = subtotal + tax_total
    return {
        "order": order,
        "items": items,
        "subtotal": subtotal,
        "tax_total": tax_total,
        "total": total,
    }

def build_receipt_html(order_id: str) -> str:
    data = get_order_with_items(order_id)
    if not data:
        return "<p>Order not found</p>"

    order = data["order"]
    items = data["items"]
    subtotal = data["subtotal"]
    tax_total = data["tax_total"]
    total = data["total"]

    # ------------------------------------------------------------
    # DOCUMENT TITLE
    # ------------------------------------------------------------
    is_paid = int(order.get("paid") or 0) == 1
    doc_title = "RECEIPT" if is_paid else "BILL"

    # ------------------------------------------------------------
    # BUSINESS PROFILE (SINGLE SOURCE OF TRUTH)
    # ------------------------------------------------------------
    profile = get_business_profile()

    shop_name = html.escape(profile.get("business_name") or "My Business")
    shop_address = html.escape(profile.get("address") or "")
    shop_phone = html.escape(profile.get("phone") or "")
    tax_number = html.escape(profile.get("vat_no") or "")
    footer = html.escape(profile.get("closing_line") or "")

    # ------------------------------------------------------------
    # CURRENCY + TAX LABEL (DISPLAY ONLY — SAFE)
    # ------------------------------------------------------------
    currency = (profile.get("currency") or "GBP").upper()

    CURRENCY_SYMBOLS = {
        "GBP": "£",
        "INR": "₹",
        "AED": "د.إ",
        "QAR": "ر.ق",
        "SAR": "﷼",
        "OMR": "﷼",
    }
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    TAX_LABELS = {
        "GBP": "VAT",
        "INR": "GST",
        "AED": "TAX",
        "QAR": "TAX",
        "SAR": "TAX",
        "OMR": "TAX",
    }
    tax_label = TAX_LABELS.get(currency, "TAX")

    created = order.get("created_at", "")
    order_label = order.get("table_no") or order.get("token") or order["id"][:8]
    short_order_id = order["id"][-6:]

    lines: List[str] = []
    lines.append("<div style='font-family:monospace;max-width:320px;text-align:center;'>")

    # ---------------- HEADER ----------------
    lines.append(
        f"<h2 style='margin:0;padding:0;font-size:22px;font-weight:bold;'>{shop_name}</h2>"
    )

    if shop_address:
        lines.append(f"<div style='font-size:12px;margin-top:2px;'>{shop_address}</div>")

    if shop_phone:
        lines.append(f"<div style='font-size:12px;'>{shop_phone}</div>")

    if tax_number:
        lines.append(
            f"<div style='font-size:12px;margin-bottom:4px;'>{tax_label}: {tax_number}</div>"
        )

    lines.append(
        f"<div style='margin-top:6px;font-size:16px;font-weight:900;'>{doc_title}</div>"
    )

    lines.append("<hr style='border-top:1px dashed #000;'/>")

    # ---------------- META ----------------
    lines.append("<div style='text-align:left;font-size:13px;'>")
    lines.append(f"<b>Order:</b> {order_label}<br>")
    lines.append(f"<b>Order ID:</b> {short_order_id}<br>")
    lines.append(f"<b>Date:</b> {created}<br>")
    lines.append("</div>")

    lines.append("<hr style='border-top:1px dashed #000;'/>")

    # ---------------- ITEMS ----------------
    lines.append("<div style='font-size:14px;text-align:left;'>")
    for it in items:
        name = html.escape(it["name"])
        qty = it["qty"]
        price = it["unit_price"]
        base = qty * price
        line_tax = base * (it["tax"] / 100)
        line_total = base + line_tax

        lines.append(
            "<div style='display:flex;justify-content:space-between;'>"
            f"<div>{name} x{qty}</div>"
            f"<div>{symbol}{line_total:.2f}</div>"
            "</div>"
        )

        if it.get("note"):
            note = html.escape(it["note"])
            lines.append(
                f"<div style='margin-left:8px;font-size:12px;color:#444;'>Note: {note}</div>"
            )

    lines.append("</div>")

    # ---------------- TOTALS ----------------
    lines.append("<hr style='border-top:1px dashed #000;'/>")

    lines.append(
        "<div style='display:flex;justify-content:space-between;font-size:14px;'>"
        f"<div>Subtotal</div><div>{symbol}{subtotal:.2f}</div></div>"
    )
    lines.append(
        "<div style='display:flex;justify-content:space-between;font-size:14px;'>"
        f"<div>{tax_label}</div><div>{symbol}{tax_total:.2f}</div></div>"
    )
    lines.append(
        "<div style='display:flex;justify-content:space-between;font-size:16px;font-weight:bold;'>"
        f"<div>Total</div><div>{symbol}{total:.2f}</div></div>"
    )

    lines.append("<hr style='border-top:1px dashed #000;'/>")

    # ---------------- CUSTOMER ----------------
    customer = order.get("customer_name") or ""
    if customer:
        lines.append(
            f"<div style='font-size:14px;text-align:left;'><b>Customer:</b> {html.escape(customer)}</div>"
        )

    if footer:
        lines.append(
            f"<div style='margin-top:10px;font-size:12px;text-align:center;'>{footer}</div>"
        )

    lines.append("</div>")

    html_body = "\n".join(lines)

    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        @media print {{
          @page {{ size: 72mm auto; margin: 4mm; }}
          .print-controls {{ display: none !important; }}
        }}
        body {{ font-family: monospace; }}
      </style>
    </head>
    <body>
      <div id="receipt">{html_body}</div>
      <div class="print-controls" style="margin-top:10px;text-align:center;">
        <button style="padding:6px 20px;font-size:14px;" onclick="window.print();">
          Print
        </button>
      </div>
    </body>
    </html>
    """

def show_bill(order_id: str):
    try:
        receipt_html = build_receipt_html(order_id)
        st.components.v1.html(receipt_html, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Failed to generate bill preview: {e}")

# -------------------------------------------------------------------
# MENU MANAGEMENT, INVENTORY, REPORTS, ADMIN, TOUCH POS
# -------------------------------------------------------------------
# ---------------- Menu Management ----------------

def menu_management():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Menu Management")

    # 🔒 ACCESS CONTROL (PERMISSION-BASED, NOT ROLE-BASED)
    if not user_can_access("Menu"):
        st.warning("🚫 You don’t have permission to access Menu.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ------------------------------------------------------------
    # CURRENCY SYMBOL (DISPLAY ONLY – SAFE)
    # ------------------------------------------------------------
    profile = get_business_profile()
    currency = (profile.get("currency") or "GBP").upper()

    CURRENCY_SYMBOLS = {
        "GBP": "£",
        "INR": "₹",
        "AED": "د.إ",
        "QAR": "ر.ق",
        "SAR": "﷼",
        "OMR": "﷼",
    }
    symbol = CURRENCY_SYMBOLS.get(currency, "£")

    # ------------------------------------------------------------
    # STATE KEYS
    # ------------------------------------------------------------
    st.session_state.setdefault("show_add_menu", False)
    st.session_state.setdefault("editing_menu_id", None)

    # ------------------------------------------------------------
    # TOP BAR
    # ------------------------------------------------------------
    cols = st.columns([3, 1])
    with cols[0]:
        q = st.text_input("Search menu item by name or category", "")
    with cols[1]:
        if st.button("➕ Add New Item"):
            st.session_state["show_add_menu"] = True
            st.session_state["editing_menu_id"] = None

    # ------------------------------------------------------------
    # LOAD ITEMS
    # ------------------------------------------------------------
    items = fetchall("SELECT * FROM menu ORDER BY category, name")
    if q:
        items = [
            i for i in items
            if q.lower() in i["name"].lower()
            or q.lower() in (i["category"] or "").lower()
        ]

    # ------------------------------------------------------------
    # ADD NEW ITEM
    # ------------------------------------------------------------
    if st.session_state["show_add_menu"]:
        with st.form("add_menu_item"):
            name = st.text_input("Name")
            category = st.text_input("Category", "Main")
            price = st.number_input("Price", min_value=0.0, format="%.2f")
            tax = st.number_input("Tax %", min_value=0.0, max_value=100.0, format="%.2f")
            stock = st.number_input("Stock", min_value=0)
            notes = st.text_area("Notes")

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Create")
            with col2:
                cancelled = st.form_submit_button("Cancel")

            if cancelled:
                st.session_state["show_add_menu"] = False
                st.rerun()

            if submitted:
                execute(
                    """
                    INSERT INTO menu (id, name, category, price, tax, stock, available, notes)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (str(uuid.uuid4()), name, category, price, float(tax), stock, 1, notes),
                )
                st.success("Menu item added")
                st.session_state["show_add_menu"] = False
                st.rerun()

    # ------------------------------------------------------------
    # EMPTY STATE
    # ------------------------------------------------------------
    if not items:
        st.info("No menu items found")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ------------------------------------------------------------
    # LIST + EDIT
    # ------------------------------------------------------------
    for item in items:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        # ---------------- VIEW MODE ----------------
        if st.session_state["editing_menu_id"] != item["id"]:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{item['name']}**")
                st.markdown(
                    f"<div class='small-muted'>"
                    f"{item['category']} • {symbol}{item['price']:.2f} • "
                    f"Tax: {item['tax']:.2f}% • Stock: {int(item['stock'])}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with c2:
                col1, col2 = st.columns(2)
                if col1.button("Edit", key=f"edit_{item['id']}"):
                    st.session_state["editing_menu_id"] = item["id"]
                    st.session_state["show_add_menu"] = False
                    st.rerun()

                if col2.button("Delete", key=f"del_{item['id']}"):
                    execute("DELETE FROM menu WHERE id=?", (item["id"],))
                    st.rerun()

        # ---------------- EDIT MODE ----------------
        else:
            with st.form(f"edit_{item['id']}"):
                name = st.text_input("Name", item["name"])
                category = st.text_input("Category", item["category"])
                price = st.number_input("Price", value=item["price"], format="%.2f")
                tax = st.number_input(
                    "Tax %",
                    value=float(item["tax"]),
                    min_value=0.0,
                    max_value=100.0,
                    format="%.2f",
                )
                stock = st.number_input("Stock", value=int(item["stock"]), min_value=0)
                available = st.checkbox("Available", value=bool(item["available"]))
                notes = st.text_area("Notes", item["notes"])

                col1, col2 = st.columns(2)
                with col1:
                    saved = st.form_submit_button("💾 Save")
                with col2:
                    cancelled = st.form_submit_button("❌ Cancel")

                if cancelled:
                    st.session_state["editing_menu_id"] = None
                    st.rerun()

                if saved:
                    execute(
                        """
                        UPDATE menu
                        SET name=?, category=?, price=?, tax=?, stock=?, available=?, notes=?
                        WHERE id=?
                        """,
                        (name, category, price, float(tax), stock, int(available), notes, item["id"]),
                    )
                    st.success("Item updated")
                    st.session_state["editing_menu_id"] = None
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Inventory Panel ----------------

def inventory_panel():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Inventory")

    items = fetchall("SELECT * FROM menu ORDER BY stock")
    if not items:
        st.info("No inventory")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    header = st.columns([2, 2, 1, 1, 3])
    header[0].markdown("**Name**")
    header[1].markdown("**Category**")
    header[2].markdown("**Stock**")
    header[3].markdown("**Price**")
    header[4].markdown("**Action**")

    for it in items:
        row = st.columns([2, 2, 1, 1, 3])
        row[0].write(it["name"])
        row[1].write(it["category"])
        row[2].write(it["stock"])
        row[3].write(it["price"])

        with row[4]:
            edit_key = f"_inv_editing_{it['id']}"
            msg_key = f"_inv_msg_{it['id']}"

            if not st.session_state.get(edit_key):
                if st.button("Edit", key=f"edit_btn_{it['id']}"):
                    st.session_state[edit_key] = True
                    st.rerun()
            else:
                with st.form(f"edit_stock_form_{it['id']}"):
                    new_stock = st.number_input(
                        "New stock",
                        min_value=0,
                        value=int(it["stock"]),
                        key=f"stock_input_{it['id']}",
                    )
                    update = st.form_submit_button("Update")
                    cancel = st.form_submit_button("Cancel")

                    if update:
                        try:
                            execute(
                                "UPDATE menu SET stock=? WHERE id=?",
                                (new_stock, it["id"]),
                            )
                            st.session_state[msg_key] = (
                                "success",
                                "Stock updated successfully",
                            )
                        except Exception as e:
                            st.session_state[msg_key] = ("error", f"Update failed: {e}")
                        del st.session_state[edit_key]
                        st.rerun()
                    elif cancel:
                        del st.session_state[edit_key]
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def order_builder():
    # ============================================================
    # 🌍 CURRENCY + TAX LABEL (DISPLAY ONLY – SAFE, GLOBAL)
    # ============================================================
    profile = get_business_profile()
    currency = (profile.get("currency") or "GBP").upper()

    CURRENCY_SYMBOLS = {
        "GBP": "£",
        "INR": "₹",
        "AED": "د.إ",
        "QAR": "ر.ق",
        "SAR": "﷼",
        "OMR": "﷼",
    }

    symbol = CURRENCY_SYMBOLS.get(currency, "£")
    tax_label = "GST" if currency == "INR" else "VAT"

    st.session_state["currency"] = currency
    st.session_state["currency_symbol"] = symbol
    st.session_state["tax_label"] = tax_label

    # ============================================================
# 🔁 AUTO REFRESH (SAFE)
# ============================================================
# try:
#     from streamlit_autorefresh import st_autorefresh
#     st_autorefresh(interval=5000, key="order_autorefresh")
# except Exception:
#     pass


    # ============================================================
    # 🔔 REALTIME BILL REQUESTS
    # ============================================================
    event = get_realtime_event()
    new_ding = False

    if event and event.get("type") == "bill_request":
        tbl = event["data"].get("table_no")
        if tbl:
            st.session_state.setdefault("_bill_alerts", {})
            if tbl not in st.session_state["_bill_alerts"]:
                st.session_state["_bill_alerts"][tbl] = time.time()
                new_ding = True
            else:
                st.session_state["_bill_alerts"][tbl] = time.time()

    now = time.time()
    for tbl, ts in list(st.session_state.get("_bill_alerts", {}).items()):
        # 30s lifetime for alert
        if now - ts > 30:
            del st.session_state["_bill_alerts"][tbl]

    for tbl in st.session_state.get("_bill_alerts", {}):
        st.markdown(
            f"""
            <div style='background:#e0f2fe;border:2px solid #3b82f6;
                        border-radius:10px;padding:10px;margin:6px 0;
                        font-size:1.2rem;font-weight:800;text-align:center;color:#000;'>
                💰 BILL REQUEST — TABLE {tbl}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if new_ding:
        st.markdown(
            """
            <audio autoplay>
                <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg">
            </audio>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # UI STYLES (UNCHANGED)
    # ============================================================
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap:14px; margin-bottom:10px; }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg,#1e3a8a,#1e40af);
        color:white!important;
        border-radius:18px;
        padding:24px 60px!important;
        font-size:2rem!important;
        font-weight:900!important;
        height:110px!important;
        border:3px solid #60a5fa;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg,#3b82f6,#1d4ed8);
        border:3px solid #93c5fd;
        transform:scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(TOUCH_POS_CSS, unsafe_allow_html=True)
    st.markdown('<div class="touch-root">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0;'>Order Management</h3>", unsafe_allow_html=True)

    menu_items = fetchall("SELECT * FROM menu WHERE available=1 ORDER BY category, name")

    # ============================================================
    # BILL PREVIEW
    # ============================================================
    bill_order_id = st.session_state.get("__bill_to_show__")
    if bill_order_id:
        if st.button("⬅️ Back"):
            st.session_state["__bill_to_show__"] = None
            st.rerun()
        show_bill(bill_order_id)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ============================================================
    # MAIN TABS
    # ============================================================
    tabs = st.tabs(["🍽️ DINE-IN", "🥡 TAKEAWAY", "📱 APPROVALS"])

    # ============================================================
    # 🍽️ DINE-IN (FIXED ORDER RESOLUTION)
    # ============================================================
    with tabs[0]:
        table_no = st.selectbox("Table", [f"T{n}" for n in range(1, 21)], key="table_select_touch")

        prev = st.session_state.get("_last_table_selected_touch")
        if prev != table_no:
            st.session_state.pop("open_table_order_id", None)
            st.session_state["_last_table_selected_touch"] = table_no

        # ✅ Prefer ACTIVE dine / approved-QR orders WITH items
        existing_with_items = fetchone("""
            SELECT o.*
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.id
            WHERE o.table_no = ?
              AND o.status IN ('pending','in_progress','ready','served')
              AND (o.order_type='dine' OR (o.order_type='qr' AND o.approved=1))
            GROUP BY o.id
            ORDER BY o.created_at DESC
            LIMIT 1
        """, (table_no,))

        # ✅ Fallback: empty dine order
        existing_dine_only = fetchone("""
            SELECT *
            FROM orders
            WHERE table_no = ?
              AND status IN ('pending','in_progress','ready','served')
              AND order_type='dine'
            ORDER BY created_at DESC
            LIMIT 1
        """, (table_no,))

        # 🚫 Pending QR (waiting approval)
        pending_qr = fetchone("""
            SELECT *
            FROM orders
            WHERE table_no = ?
              AND order_type='qr'
              AND COALESCE(approved,0)=0
              AND status IN ('pending','in_progress')
            LIMIT 1
        """, (table_no,))

        order_id = None  # ✅ important initialisation

        if existing_with_items:
            # use dine / approved-QR order that has items
            order_id = existing_with_items["id"]

        elif existing_dine_only and not pending_qr:
            # dine exists and no pending QR → normal behaviour
            order_id = existing_dine_only["id"]

        elif pending_qr:
            # ❗ Show message but DO NOT return from function
            st.warning(
                f"Table {table_no} has a pending QR order awaiting approval.\n\n"
                "Please approve it in the 📱 APPROVALS tab."
            )
            # do NOT create new dine order; just block editing here

        else:
            # no dine / approved-QR / pending-QR → create fresh dine order
            order_id = create_order("dine", table_no, None)

        # Only render cart + menu if we actually have an order_id
        if order_id:
            st.session_state["open_table_order_id"] = order_id

            left, right = st.columns([3, 2])
            with left:
                render_touch_category_bar(menu_items, context="dine")
                render_touch_item_grid(order_id, menu_items, context="dine")
            with right:
                render_touch_cart(order_id, context="dine")
        else:
            # we are in "pending QR" scenario
            st.info("Waiting for QR order approval. Go to 📱 APPROVALS to process it.")

    # ============================================================
    # 🥡 TAKEAWAY (UNCHANGED)
    # ============================================================
    with tabs[1]:
        if st.button("🆕 New Takeaway Order"):
            tid = create_order("takeaway", None, "Walk-in", paid=False)
            st.session_state["open_takeaway_order_id"] = tid
            push_realtime_event("order_created", {"order_id": tid})
            st.rerun()

        tk_id = st.session_state.get("open_takeaway_order_id")
        if tk_id:
            left, right = st.columns([3, 2])
            with left:
                render_touch_category_bar(menu_items, context="tky")
                render_touch_item_grid(tk_id, menu_items, context="tky")
            with right:
                render_touch_cart(tk_id, context="tky")
        else:
            st.info("Click **New Takeaway Order** to begin.")

    # ============================================================
    # 📱 APPROVALS
    # ============================================================
    with tabs[2]:
        approver_panel()

    st.markdown("</div>", unsafe_allow_html=True)


def approver_panel():
    """
    📱 QR Order Approvals
    ----------------------
    • Shows all QR orders waiting for approval
    • Approve  -> converts to DINE-IN order for that table
    • Reject   -> marks order as rejected (items kept for audit)
    """
    st.markdown("### 📱 QR Order Approvals")

    # Optional: basic access control – piggyback on Orders permission
    if not user_can_access("Orders"):
        st.warning("🚫 You don’t have permission to manage approvals.")
        return

    # ------------------------------------------------------------
    # LOAD PENDING QR ORDERS  (MUST MATCH order_builder pending_qr)
    # ------------------------------------------------------------
    pending_qr = fetchall(
        """
        SELECT
            o.id,
            o.table_no,
            o.customer_name,
            o.status,
            o.created_at,
            COALESCE(SUM(oi.qty * oi.unit_price), 0) AS subtotal,
            COALESCE(SUM(oi.qty * oi.unit_price * (oi.tax / 100.0)), 0) AS tax,
            COALESCE(SUM(oi.qty * oi.unit_price * (1 + oi.tax / 100.0)), 0) AS grand_total
        FROM orders o
        LEFT JOIN order_items oi ON oi.order_id = o.id
        WHERE o.order_type = 'qr'
          AND COALESCE(o.approved, 0) = 0
          AND o.status IN ('pending', 'in_progress')
        GROUP BY
            o.id,
            o.table_no,
            o.customer_name,
            o.status,
            o.created_at
        ORDER BY o.created_at ASC
        """
    )

    if not pending_qr:
        st.info("✅ No QR orders pending approval.")
        return

    # ------------------------------------------------------------
    # BUSINESS CURRENCY SYMBOL (for display only)
    # ------------------------------------------------------------
    profile = get_business_profile()
    currency = (profile.get("currency") or "GBP").upper()
    CURRENCY_SYMBOLS = {
        "GBP": "£",
        "INR": "₹",
        "AED": "د.إ",
        "QAR": "ر.ق",
        "SAR": "﷼",
        "OMR": "﷼",
    }
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    # ------------------------------------------------------------
    # RENDER EACH PENDING QR ORDER
    # ------------------------------------------------------------
    for o in pending_qr:
        st.markdown("---")
        st.markdown(
            f"#### 🧾 Table **{o['table_no'] or '-'}** — Order **{o['id'][:8]}**"
        )
        st.markdown(
            f"- Status: **{(o['status'] or '').upper()}**  \n"
            f"- Customer: **{o['customer_name'] or 'QR Guest'}**  \n"
            f"- Created: `{o['created_at']}`"
        )

        # Load line items for this order
        items = fetchall(
            """
            SELECT name, qty, unit_price, tax
            FROM order_items
            WHERE order_id = ?
            ORDER BY created_at
            """,
            (o["id"],),
        )

        if items:
            st.markdown("**Items:**")
            for it in items:
                base = it["qty"] * it["unit_price"]
                tax_val = base * (it["tax"] / 100.0)
                line_total = base + tax_val
                st.markdown(
                    f"- {it['name']} ×{it['qty']} — "
                    f"{symbol}{line_total:.2f} "
                    f"(_{it['tax']:.1f}% tax_)"
                )
        else:
            st.warning("No items found on this order (check QR flow).")

        st.markdown(
            f"**Subtotal:** {symbol}{float(o['subtotal']):.2f}  \n"
            f"**Tax:** {symbol}{float(o['tax']):.2f}  \n"
            f"**Total:** {symbol}{float(o['grand_total']):.2f}"
        )

        col_a, col_b = st.columns(2)

        # ---------------- APPROVE ----------------
        with col_a:
            if st.button(
                "✅ Approve (Send to Dine-In)",
                key=f"approve_{o['id']}",
                width="stretch",
            ):
                # 🔁 Convert QR → DINE-IN + mark approved
                execute(
                    """
                    UPDATE orders
                    SET approved = 1,
                        order_type = 'dine',
                        status = 'in_progress',
                        updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (o["id"],),
                )

                # Optional realtime event
                push_realtime_event(
                    "qr_approved",
                    {
                        "order_id": o["id"],
                        "table_no": o["table_no"],
                    },
                )

                st.success(
                    f"✅ QR order {o['id'][:8]} approved and moved to DINE-IN for table {o['table_no']}."
                )
                st.rerun()

        # ---------------- REJECT ----------------
        with col_b:
            if st.button(
                "❌ Reject Order",
                key=f"reject_{o['id']}",
                width="stretch",
            ):
                execute(
                    """
                    UPDATE orders
                    SET approved = 0,
                        status = 'rejected',
                        updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    (o["id"],),
                )

                push_realtime_event(
                    "qr_rejected",
                    {
                        "order_id": o["id"],
                        "table_no": o["table_no"],
                    },
                )

                st.warning(
                    f"🚫 QR order {o['id'][:8]} marked as REJECTED. "
                    "Customer will see 'Item not available' on refresh."
                )
                st.rerun()


# ---------------- Payments Panel ----------------

def payments_panel():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Payments")

    # 🔒 Access control
    if not user_can_access("Payments"):
        st.warning("🚫 You don’t have permission to access this page.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ------------------------------------------------------
    # BUSINESS CURRENCY (DISPLAY ONLY)
    # ------------------------------------------------------
    profile = get_business_profile()
    currency = (profile.get("currency") or "GBP").upper()

    CURRENCY_SYMBOLS = {
        "GBP": "£",
        "INR": "₹",
        "AED": "د.إ",
        "QAR": "ر.ق",
        "SAR": "﷼",
        "OMR": "﷼",
    }
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    # ------------------------
    # REFRESH
    # ------------------------
    if st.button("🔄 Refresh Pending Payments"):
        st.rerun()

    # ------------------------
    # UNPAID ORDERS
    # ------------------------
    unpaid = fetchall(
        """
        SELECT id, order_type, table_no, token, total
        FROM orders
        WHERE paid=0
        ORDER BY created_at DESC
        """
    )

    if not unpaid:
        st.info("No unpaid orders.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    order_labels = [
        f"{o['id'][:8]} | {o['order_type']} | "
        f"{o.get('table_no') or o.get('token') or ''} | "
        f"{symbol}{o['total']:.2f}"
        for o in unpaid
    ]

    sel = st.selectbox("Select order to pay", order_labels)
    sel_prefix = sel.split("|")[0].strip()
    order = next((o for o in unpaid if o["id"].startswith(sel_prefix)), None)

    if not order:
        st.error("Invalid order selection.")
        return

    amount_due = float(order["total"] or 0.0)

    st.markdown(f"**Order**: {order['id'][:8]}")
    st.markdown(f"**Total Payable**: {symbol}{amount_due:.2f}")

    # ------------------------
    # PAYMENT METHOD
    # ------------------------
    method = st.selectbox(
        "Payment Method",
        [
            "Cash",
            "Card (Machine / S700 Manual)",
            "Online Card (Stripe Checkout)",
        ],
    )

    amt = st.number_input(
        "Amount received",
        value=amount_due,
        min_value=0.0,
        format="%.2f",
    )

    # ------------------------
    # CONFIRM PAYMENT
    # ------------------------
    if st.button("✅ Confirm & Record Payment"):
        if amt <= 0:
            st.error("Amount must be greater than zero.")
        else:
            execute(
                """
                INSERT INTO payments (id, order_id, method, amount, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    order["id"],
                    method,
                    amt,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            execute(
                "UPDATE orders SET paid=1 WHERE id=?",
                (order["id"],),
            )

            push_realtime_event(
                "order_paid",
                {
                    "order_id": order["id"],
                    "amount": float(amt),
                    "method": method,
                },
            )

            st.success("💰 Payment recorded successfully!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # 📅 PAYMENT HISTORY (TAX REMOVED — CLEAN & CORRECT)
    # ============================================================
    st.markdown("### 📅 Payment History")

    selected_date = st.date_input("Select date", value=date.today())
    start_dt = datetime.combine(selected_date, datetime.min.time()).isoformat()
    end_dt = datetime.combine(selected_date, datetime.max.time()).isoformat()

    completed = fetchall(
        """
        SELECT
            o.id,
            o.order_type,
            o.table_no,
            o.token,
            p.method,
            p.created_at,
            p.amount
        FROM orders o
        JOIN payments p ON o.id = p.order_id
        WHERE p.created_at BETWEEN ? AND ?
        ORDER BY p.created_at DESC
        """,
        (start_dt, end_dt),
    )

    if completed:
        df = pd.DataFrame({
            "Bill No": [o["id"][:8] for o in completed],
            "Order Type": [o["order_type"].upper() for o in completed],
            "Table/Token": [o["table_no"] or o["token"] for o in completed],
            "Payment Type": [o["method"] for o in completed],
            f"Paid Amount ({symbol})": [float(o["amount"]) for o in completed],
            "Payment Time": [o["created_at"] for o in completed],
        })

        st.dataframe(df, width="stretch")
    else:
        st.info("No payments found for selected date.")


# ---------------- Reports Panel ----------------

def reports_panel():
    import io

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Reports")

    # ------------------------------------------------------
    # ACCESS CONTROL (PERMISSION-BASED ✅)
    # ------------------------------------------------------
    if not user_can_access("Reports"):
        st.warning("🚫 You don’t have permission to access Reports.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ------------------------------------------------------
    # BUSINESS CURRENCY + TAX LABEL (SINGLE SOURCE OF TRUTH)
    # ------------------------------------------------------
    profile = get_business_profile()
    currency = (profile.get("currency") or "GBP").upper()

    CURRENCY_SYMBOLS = {
        "GBP": "£",
        "INR": "₹",
        "AED": "د.إ",
        "QAR": "ر.ق",
        "SAR": "﷼",
        "OMR": "﷼",
    }
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    TAX_LABELS = {
        "GBP": "VAT",
        "INR": "GST",
        "AED": "TAX",
        "QAR": "TAX",
        "SAR": "TAX",
        "OMR": "TAX",
    }
    tax_label = TAX_LABELS.get(currency, "TAX")

    # ------------------------------------------------------
    # DATE RANGE
    # ------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date.today())
    with col2:
        end_date = st.date_input("End date", value=date.today())

    start_dt = datetime.combine(start_date, datetime.min.time()).isoformat()
    end_dt = datetime.combine(end_date, datetime.max.time()).isoformat()

    # ------------------------------------------------------
    # PAYMENTS = SOURCE OF TRUTH
    # TAX = CALCULATED FROM ORDER ITEMS ✅
    # ------------------------------------------------------
    rows = fetchall(
        """
        SELECT
            o.id,
            o.order_type,
            o.table_no,
            o.token,
            p.method AS payment_type,
            p.created_at AS payment_time,
            p.amount AS paid_amount,

            (
                SELECT COALESCE(
                    SUM(oi.qty * oi.unit_price * (oi.tax / 100.0)), 0
                )
                FROM order_items oi
                WHERE oi.order_id = o.id
            ) AS tax_amount

        FROM orders o
        JOIN payments p ON o.id = p.order_id
        WHERE p.created_at BETWEEN ? AND ?
        ORDER BY p.created_at DESC
        """,
        (start_dt, end_dt),
    )

    if not rows:
        st.info("No payments found for selected dates.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ------------------------------------------------------
    # DATAFRAME (EXPORT-SAFE)
    # ------------------------------------------------------
    df = pd.DataFrame({
        "Order ID": [r["id"] for r in rows],
        "Bill No": [r["id"][:8] for r in rows],
        "Order Type": [r["order_type"].upper() for r in rows],
        "Table/Token": [r["table_no"] or r["token"] or "-" for r in rows],
        "Payment Type": [r["payment_type"] for r in rows],
        "Payment Time": [r["payment_time"] for r in rows],
        f"{tax_label} Amount ({symbol})": [float(r["tax_amount"]) for r in rows],
        f"Paid Amount ({symbol})": [float(r["paid_amount"]) for r in rows],
    })

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------
    total_tax = df[f"{tax_label} Amount ({symbol})"].sum()
    total_paid = df[f"Paid Amount ({symbol})"].sum()

    c1, c2 = st.columns(2)
    c1.metric(f"Total {tax_label}", f"{symbol}{total_tax:.2f}")
    c2.metric("Total Sales", f"{symbol}{total_paid:.2f}")

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------
    st.dataframe(df, width="stretch")

    # ------------------------------------------------------
    # CSV EXPORT
    # ------------------------------------------------------
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇️ Download Report (CSV)",
        csv_buffer.getvalue(),
        file_name=f"sales_tax_report_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------
    # EXCEL EXPORT
    # ------------------------------------------------------
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sales + Tax")

    st.download_button(
        "⬇️ Download Report (Excel)",
        excel_buffer.getvalue(),
        file_name=f"sales_tax_report_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("</div>", unsafe_allow_html=True)

import qrcode
import os


def get_business_profile():
    """
    Fetch single-row business profile for receipts/bills
    Includes currency + lock status
    """
    row = fetchone(
        """
        SELECT
            business_name,
            address,
            phone,
            vat_no,
            closing_line,
            currency,
            currency_locked
        FROM business_profile
        WHERE id = 1
        """
    )

    if not row:
        return {
            "business_name": "",
            "address": "",
            "phone": "",
            "vat_no": "",
            "closing_line": "",
            "currency": "GBP",
            "currency_locked": 0
        }

    return row


def save_business_profile(
    business_name,
    address,
    phone,
    vat_no,
    closing_line,
    currency=None,
    lock_currency=False
):
    """
    Save receipt/bill business details (single-row update)

    ✅ Safe extension:
    - Existing calls still work (currency optional)
    - Currency can be set once and locked (lock_currency=True)
    """

    # Normal update (always safe)
    execute(
        """
        UPDATE business_profile
        SET business_name = ?,
            address = ?,
            phone = ?,
            vat_no = ?,
            closing_line = ?
        WHERE id = 1
        """,
        (business_name, address, phone, vat_no, closing_line)
    )

    # Optional: currency set + lock (one-time)
    if currency:
        row = fetchone("SELECT currency_locked, currency FROM business_profile WHERE id=1")
        locked = int((row or {}).get("currency_locked") or 0)

        # Only allow setting currency if not locked yet
        if locked == 0:
            if lock_currency:
                execute(
                    """
                    UPDATE business_profile
                    SET currency = ?,
                        currency_locked = 1
                    WHERE id = 1
                    """,
                    (currency,)
                )
            else:
                execute(
                    """
                    UPDATE business_profile
                    SET currency = ?
                    WHERE id = 1
                    """,
                    (currency,)
                )


# ---------------- Admin Panel ----------------
def admin_panel():
    """
    🧭 Admin Settings — Centralized User & Access Control
    -------------------------------------------------------
    • Only superadmin can open this panel
    • Create / Edit / Delete users
    • Assign which modules each user can access
    • Configure Receipt / Bill Business Details
    • Currency can be set ONCE and locked permanently
    • Change Superadmin username & password
    • Generate HMAC-secure QR links for tables
    • Download QR PNG files (ZIP)
    -------------------------------------------------------
    """
    import json, urllib.parse
    import streamlit as st
    import pandas as pd
    import io
    import zipfile
    import os

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Admin Settings")

    # ---------------- ACCESS ----------------
    if st.session_state["user"]["role"] != "superadmin":
        st.warning("🚫 Restricted: Only Superadmin can access Admin Settings.")
        return

    current_user = st.session_state["user"]

    # ============================================================
    # 🧾 RECEIPT / BILL BUSINESS DETAILS
    # ============================================================
    with st.expander("🧾 Receipt / Bill Business Details", expanded=False):
        profile = get_business_profile()

        business_name = st.text_input("Business Name", value=profile.get("business_name") or "")
        address = st.text_area("Address", value=profile.get("address") or "")
        phone = st.text_input("Phone Number", value=profile.get("phone") or "")
        vat_no = st.text_input("VAT / GST Number", value=profile.get("vat_no") or "")
        closing_line = st.text_input("Receipt Closing Line", value=profile.get("closing_line") or "")

        SUPPORTED_CURRENCIES = ["GBP", "INR", "AED", "QAR", "SAR", "OMR"]
        currency_locked = int(profile.get("currency_locked") or 0)
        current_currency = profile.get("currency") or "GBP"

        currency = st.selectbox(
            "Business Currency (can be set only once)",
            SUPPORTED_CURRENCIES,
            index=SUPPORTED_CURRENCIES.index(current_currency),
            disabled=bool(currency_locked),
        )

        if currency_locked:
            st.info("🔒 Currency is locked and cannot be changed.")
        else:
            st.warning("⚠️ Currency can be set ONLY ONCE.")

        if st.button("💾 Save Receipt Details"):
            if currency_locked == 0:
                save_business_profile(
                    business_name, address, phone, vat_no, closing_line,
                    currency=currency, lock_currency=True
                )
            else:
                save_business_profile(
                    business_name, address, phone, vat_no, closing_line
                )
            st.success("✅ Business & receipt details saved")
            st.rerun()

    # ============================================================
    # 🔐 CHANGE SUPERADMIN CREDENTIALS
    # ============================================================
    with st.expander("🔐 Change My Superadmin Login", expanded=False):
        st.warning("⚠️ This will log you out after saving.")

        new_username = st.text_input("New Username", value=current_user["username"])
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if st.button("🔁 Update My Login"):
            if not current_password:
                st.error("Current password is required.")
            elif new_password and new_password != confirm_password:
                st.error("New password and confirmation do not match.")
            else:
                db_user = fetchone("SELECT * FROM users WHERE id=?", (current_user["id"],))
                if not db_user or not verify_password(current_password, db_user["password_hash"]):
                    st.error("❌ Current password is incorrect.")
                else:
                    updates, params = [], []

                    if new_username and new_username != current_user["username"]:
                        updates.append("username=?")
                        params.append(new_username)

                    if new_password:
                        updates.append("password_hash=?")
                        params.append(hash_password(new_password))

                    if updates:
                        params.append(current_user["id"])
                        execute(
                            f"UPDATE users SET {', '.join(updates)} WHERE id=?",
                            tuple(params)
                        )
                        st.success("✅ Login updated. Please log in again.")
                        st.session_state.clear()
                        st.rerun()
                    else:
                        st.info("No changes detected.")

    # ============================================================
    # USER MANAGEMENT
    # ============================================================
    MODULE_LIST = [
        "Home", "Orders", "KDS", "Takeaway Status",
        "Inventory", "Menu", "Payments", "Reports", "Admin"
    ]

    # ---------------- CREATE USER ----------------
    st.markdown("## ➕ Create New User")
    with st.form("create_user_form"):
        username = st.text_input("Username")
        full_name = st.text_input("Full Name")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["superadmin", "manager", "cashier", "kitchen"])

        st.markdown("#### Select Modules:")
        module_access = {}
        cols = st.columns(3)
        for i, m in enumerate(MODULE_LIST):
            with cols[i % 3]:
                module_access[m] = st.checkbox(m, value=(role == "superadmin"))

        if st.form_submit_button("Create User"):
            if not username or not password:
                st.error("⚠️ Username and password required.")
            else:
                allowed = MODULE_LIST.copy() if role == "superadmin" else [m for m, v in module_access.items() if v]
                create_user(username, password, full_name, role, allowed)
                st.success(f"✅ User '{username}' created.")
                st.rerun()

    # ---------------- MANAGE USERS ----------------
    st.markdown("---")
    st.markdown("## 👥 Manage Users")

    users = fetchall(
        "SELECT id, username, full_name, role, allowed_modules FROM users ORDER BY username"
    )

    for u in users:
        st.markdown("---")
        st.markdown(f"### 👤 {u['username']} ({u['role']})")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**Name:** {u['full_name']}")
            st.write(f"**Role:** {u['role']}")

        with col2:
            try:
                current_modules = json.loads(u["allowed_modules"] or "[]")
            except Exception:
                current_modules = []
            st.write("**Modules:**")
            st.write(", ".join(current_modules) if current_modules else "None")

        with col3:
            if u["role"] == "superadmin":
                count_sa = fetchone("SELECT COUNT(*) AS c FROM users WHERE role='superadmin'")["c"]
                if count_sa <= 1 or u["id"] == current_user["id"]:
                    st.caption("Cannot delete this superadmin")
                else:
                    if st.button("❌ Delete", key=f"del_{u['id']}"):
                        execute("DELETE FROM users WHERE id=?", (u["id"],))
                        st.success("User deleted.")
                        st.rerun()
            else:
                if st.button("❌ Delete", key=f"del_{u['id']}"):
                    execute("DELETE FROM users WHERE id=?", (u["id"],))
                    st.success("User deleted.")
                    st.rerun()

        with st.expander("✏️ Edit User Permissions"):
            cols_edit = st.columns(3)
            edit_modules = {}
            for i, m in enumerate(MODULE_LIST):
                with cols_edit[i % 3]:
                    edit_modules[m] = st.checkbox(
                        m, value=(m in current_modules), key=f"edit_{u['id']}_{m}"
                    )

            if st.button("💾 Save Changes", key=f"save_{u['id']}"):
                new_allowed = MODULE_LIST.copy() if u["role"] == "superadmin" else [m for m, v in edit_modules.items() if v]
                execute(
                    "UPDATE users SET allowed_modules=? WHERE id=?",
                    (json.dumps(new_allowed), u["id"]),
                )
                st.success("✅ Permissions updated.")
                st.rerun()

    # ============================================================
    # 🔗 SECURE QR GENERATOR (LINK + PNG + ZIP)
    # ============================================================
    st.markdown("---")
    with st.expander("🧾 Generate Secure QR Links for Tables"):
        table_list_input = st.text_input("Table Numbers (e.g. T1, T2, T3)")

        if st.button("🔗 Generate Secure Links & QR Codes"):
            base_url = st.secrets.get("PUBLIC_QR_BASE_URL", "http://localhost:8501")
            tables = [t.strip().upper() for t in table_list_input.split(",") if t.strip()]

            rows = []
            qr_files = []

            for t in tables:
                sig = generate_qr_sig(t)
                url = f"{base_url}?qr=1&table={urllib.parse.quote(t)}&sig={sig}"
                qr_path = generate_qr_from_link(url, f"TABLE_{t}")

                rows.append({
                    "Table": t,
                    "Secure URL": url,
                    "QR File": os.path.basename(qr_path)
                })
                qr_files.append(qr_path)

            st.dataframe(pd.DataFrame(rows), width="stretch")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in qr_files:
                    zf.write(f, arcname=os.path.basename(f))

            st.download_button(
                "⬇️ Download All QR Codes (PNG ZIP)",
                zip_buffer.getvalue(),
                file_name="table_qr_codes.zip",
                mime="application/zip"
            )



def takeaway_status_panel():
    """
    Read-only Takeaway Status screen:
    ---------------------------------
    - Reflects KITCHEN truth only
    - ACTIVE orders:
        • in_progress
        • ready
    - CLOSED / PAID:
        • shown for TODAY only (history)
    - Safe across midnight
    - Search by TOKEN NUMBER only
    - ❗ Now hides takeaway orders that have no items
    """
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Takeaway Status")

    # ---- Filters ----
    col1, col2 = st.columns([2, 1])
    with col1:
        q = (
            st.text_input(
                "Search by token number (e.g. A009)",
                key="tky_status_search"
            )
            .strip()
            .upper()
        )

    with col2:
        include_closed = st.checkbox(
            "Include fully closed / paid",
            value=True,
            key="tky_status_include_closed"
        )

    # ------------------------------------------------------------
    # ✅ ACTIVE = in_progress OR ready
    # ✅ CLOSED = paid / closed (TODAY only)
    # ✅ EXCLUDE EMPTY ORDERS (no order_items)
    # ------------------------------------------------------------
    if include_closed:
        status_clause = """
        AND (
            o.status IN ('in_progress','ready')
            OR (
                o.status IN ('closed','paid')
                AND DATE(o.created_at) = DATE('now','localtime')
            )
        )
        """
    else:
        status_clause = "AND o.status IN ('in_progress','ready')"

    sql = f"""
        SELECT o.*
        FROM orders o
        WHERE o.order_type = 'takeaway'
        {status_clause}
          AND EXISTS (
              SELECT 1
              FROM order_items oi
              WHERE oi.order_id = o.id
          )
    """
    params = []

    if q:
        sql += " AND UPPER(COALESCE(o.token,'')) LIKE ?"
        params.append(f"%{q}%")

    sql += " ORDER BY o.created_at DESC"

    orders = fetchall(sql, tuple(params))

    if not orders:
        st.info("No takeaway orders found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ------------------------------------------------------------
    # DISPLAY EACH TAKEAWAY ORDER
    # ------------------------------------------------------------
    for o in orders:
        token = o.get("token") or "N/A"
        status_raw = (o.get("status") or "").upper()
        total = float(o.get("total") or 0.0)

        # 🎨 STATUS COLOR LOGIC (KITCHEN-CENTRIC)
        if status_raw in ("CLOSED", "PAID"):
            status_html = "<span style='color:#16a34a;font-weight:900;'>CLOSED</span>"
        elif status_raw == "READY":
            status_html = "<span style='color:#2563eb;font-weight:900;'>READY (COLLECT)</span>"
        elif status_raw == "IN_PROGRESS":
            status_html = "<span style='color:#dc2626;font-weight:900;'>IN PROGRESS</span>"
        else:
            status_html = status_raw

        st.markdown(
            f"""
            <div style="margin-top:8px;margin-bottom:4px;">
              <b>Token:</b> {token} &nbsp;|&nbsp;
              <b>Status:</b> {status_html} &nbsp;|&nbsp;
              <b>Total:</b> £{total:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Items + KDS status (READ-ONLY) ----
        items = fetchall(
            """
            SELECT
                oi.*,
                COALESCE(ks.status, 'pending') AS kds_status
            FROM order_items oi
            LEFT JOIN kds_items_status ks
                ON ks.order_item_id = oi.id
            WHERE oi.order_id = ?
            ORDER BY oi.created_at
            """,
            (o["id"],)
        )

        if not items:
            st.markdown(
                "<div style='font-size:0.85rem;color:#9ca3af;'>No items found.</div>",
                unsafe_allow_html=True,
            )
        else:
            df = pd.DataFrame(items)
            df["tax_amount"] = df["qty"] * df["unit_price"] * (df["tax"] / 100)
            df["line_total"] = df["qty"] * df["unit_price"] + df["tax_amount"]

            st.dataframe(
                df[
                    [
                        "name",
                        "qty",
                        "unit_price",
                        "tax",
                        "tax_amount",
                        "line_total",
                        "kds_status",
                    ]
                ].rename(
                    columns={
                        "unit_price": "Price",
                        "tax": "Tax %",
                        "tax_amount": "Tax Amt",
                        "line_total": "Line Total",
                        "kds_status": "KDS Status",
                    }
                ),
                width="stretch",
            )

        st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)


def generate_bill_html(order_id: str) -> str:
    order = fetchone("SELECT * FROM orders WHERE id=?", (order_id,))
    if not order:
        return "<pre>Order not found</pre>"

    items = fetchall(
        "SELECT * FROM order_items WHERE order_id=? ORDER BY created_at",
        (order_id,)
    )

    # ------------------------------------------------------------
    # BUSINESS PROFILE (SAFE – DISPLAY ONLY)
    # ------------------------------------------------------------
    profile = get_business_profile()

    lines = []

    # ============================================================
    # BUSINESS HEADER (OPTIONAL)
    # ============================================================
    business_name = (profile.get("business_name") or "").strip()
    address = (profile.get("address") or "").strip()
    phone = (profile.get("phone") or "").strip()
    vat_no = (profile.get("vat_no") or "").strip()

    if business_name:
        lines.append(business_name.upper())
    if address:
        lines.append(address)
    if phone:
        lines.append(f"Ph: {phone}")
    if vat_no:
        lines.append(f"VAT: {vat_no}")

    if any([business_name, address, phone, vat_no]):
        lines.append("------------------------------")

    # ============================================================
    # KOT CONTENT (NO PRICE / NO CURRENCY)
    # ============================================================
    lines.append("====== KITCHEN ORDER ======")
    lines.append(f"ORDER: {order_id[:8]}")
    lines.append(f"TYPE : {order['order_type'].upper()}")
    lines.append(f"TBL/TKN: {order['table_no'] or order['token'] or 'N/A'}")
    lines.append("------------------------------")

    for it in items:
        lines.append(f"{it['name']} x{it['qty']}")
        if it.get("note"):
            lines.append(f"  * {it['note']}")

    lines.append("------------------------------")
    lines.append(f"Printed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")

    closing_line = (profile.get("closing_line") or "").strip()
    if closing_line:
        lines.append(closing_line)

    lines.append("Que-POS Kitchen Ticket")
    lines.append("\n\n")  # feed for cutter

    return (
        "<pre style='font-family: monospace; font-size: 14px;'>"
        + "\n".join(lines) +
        "</pre>"
    )

import streamlit.components.v1 as components

import streamlit.components.v1 as components
from datetime import datetime

def print_ticket_via_component(order_id: str, only_items=None):
    """
    Kitchen ticket printing helper.

    - If only_items is None  → print FULL order (all items).
    - If only_items is a list of order_item IDs → print ONLY those items,
      with a 'New Items' style header.
    """

    # Fetch order header
    order = fetchone("SELECT * FROM orders WHERE id=?", (order_id,))
    if not order:
        html_body = "<pre>Order not found</pre>"
    else:
        # Decide which items to print
        if only_items:
            # Print only selected items
            placeholders = ",".join(["?"] * len(only_items))
            items = fetchall(
                f"SELECT * FROM order_items WHERE id IN ({placeholders}) ORDER BY created_at",
                tuple(only_items),
            )
            title = "KITCHEN ORDER (New Items)"
        else:
            # Full order print
            items = fetchall(
                "SELECT * FROM order_items WHERE order_id=? ORDER BY created_at",
                (order_id,),
            )
            title = "KITCHEN ORDER"

        # Build lines
        lines = []
        lines.append(f"====== {title} ======")
        lines.append(f"ORDER: {order_id[:8]}")
        lines.append(f"TYPE : {order['order_type'].upper()}")
        lines.append(f"TBL/TKN: {order['table_no'] or order['token'] or 'N/A'}")
        lines.append("------------------------------")

        if not items:
            lines.append("(No items)")
        else:
            for it in items:
                nm = it["name"]
                qty = it["qty"]
                lines.append(f"{nm} x{qty}")
                if it.get("note"):
                    lines.append(f"  * {it['note']}")  # indented note

        lines.append("------------------------------")
        dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"Printed: {dt}")
        lines.append("Que-POS Kitchen Ticket")
        lines.append("\n\n")  # space for cutting

        html_body = (
            "<pre style='font-family: monospace; font-size: 14px;'>"
            + "\n".join(lines) +
            "</pre>"
        )

    full_html = f"""
    <html>
    <head>
        <meta charset='UTF-8' />
        <style>
            body {{
                font-family: monospace;
                padding: 10px;
                margin: 0;
            }}
        </style>
    </head>
    <body onload="window.print();">
        {html_body}
    </body>
    </html>
    """

    # small invisible component that triggers browser print
    components.html(full_html, height=20, scrolling=False)

# ==========================================================
#   KDS PRINT TRACKING HELPERS (METHOD 3 – item ID tracking)
# ==========================================================

def _kds_get_log(order_id: str):
    """Return print log for this order."""
    st.session_state.setdefault("_kds_print_log", {})
    return st.session_state["_kds_print_log"].setdefault(order_id, {
        "full_print_done": False,
        "printed_items": set()
    })

def _kds_mark_printed(order_id: str, item_ids: list, full=False):
    """Mark item IDs as printed."""
    log = _kds_get_log(order_id)
    if full:
        log["full_print_done"] = True
    log["printed_items"].update(item_ids)


# ---------------- KDS VIEW ----------------
def kds_view():
    """
    🍳 Unified Kitchen Display System (KDS)
    --------------------------------------------
    • Shows ONLY active kitchen orders
    • Universal for Dine-in / Takeaway / Approved-QR
    • Kitchen statuses ONLY:
        - in_progress
        - ready
    • Item-level tracking (pending → done)
    • Auto-print + sound alerts preserved
    • NO order closing here (FOH responsibility)
    --------------------------------------------
    """
    import time
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Kitchen Display (KDS)")

    # ---------------- Auto-refresh ----------------
    try:
        from streamlit_autorefresh import st_autorefresh
        interval = st.selectbox(
            "Auto-refresh every (s)", [0, 3, 5, 10, 15, 30], index=3
        )
        if interval and interval > 0:
            st_autorefresh(interval * 1000, key="kds_refresh")
    except Exception:
        st.info("Install streamlit-autorefresh for auto-refresh")

    enable_sound = st.checkbox("Enable sound alerts for new items", value=True)

    # Auto-print toggle
    st.session_state.setdefault("kds_auto_print", False)
    auto_print_enabled = st.checkbox(
        "Enable Auto-Print", value=st.session_state["kds_auto_print"]
    )
    st.session_state["kds_auto_print"] = auto_print_enabled

    # ======================================================
    # 🔒 LOAD ONLY VALID KITCHEN ORDERS (FINAL FIX)
    # ======================================================
    orders_raw = fetchall("""
        SELECT *
        FROM orders
        WHERE status IN ('in_progress','ready')
          AND (
                order_type != 'qr'
                OR approved = 1
              )
        ORDER BY created_at ASC
    """)

    if not orders_raw:
        st.info("No active kitchen orders.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ======================================================
    # 🎨 CSS Styling (UNCHANGED)
    # ======================================================
    st.markdown("""
    <style>
    .status-in_progress {
        background: #d97706 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin-bottom: 6px !important;
        font-weight: 700 !important;
    }
    .status-ready {
        background: #2563eb !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin-bottom: 6px !important;
        font-weight: 700 !important;
    }
    .status-done {
        background: #15803D !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin-bottom: 6px !important;
        font-weight: 800 !important;
        opacity: 0.85 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    auto_print_queue = []
    any_new_items = False

    # ======================================================
    # DISPLAY EACH ORDER
    # ======================================================
    for o in orders_raw:
        order_id = o["id"]

        items = fetchall("""
            SELECT oi.*,
                   COALESCE(ks.status,'pending') AS kds_status
            FROM order_items oi
            LEFT JOIN kds_items_status ks
              ON ks.order_item_id = oi.id
            WHERE oi.order_id = ?
            ORDER BY oi.created_at
        """, (order_id,))

        if not items:
            continue

        # ✅ if ALL items are ready/done → hide from KDS (your existing behavior)
        all_done_for_kds = all(
            (item["kds_status"] or "").lower() in ("ready", "done")
            for item in items
        )
        if all_done_for_kds:
            continue

        # ✅ Preserve your print tracking logic
        log = _kds_get_log(order_id)
        printed_ids = log["printed_items"]

        current_ids = {it["id"] for it in items}
        new_item_ids = current_ids - printed_ids

        if new_item_ids:
            any_new_items = True
            if auto_print_enabled:
                auto_print_queue.append((order_id, list(new_item_ids), False))

        if not log["full_print_done"] and auto_print_enabled:
            auto_print_queue.append((order_id, [it["id"] for it in items], True))

        order_label = (
            f"{o['order_type'].capitalize()} • "
            f"{o.get('table_no') or o.get('token') or 'No Ref'} • "
            f"Order {order_id[:8]}"
        )

        st.markdown(f"<b>{order_label}</b>", unsafe_allow_html=True)

        # ✅ Print button restored exactly
        if st.button("🖨️ Print Ticket", key=f"print_{order_id}"):
            if not log["full_print_done"]:
                print_ticket_via_component(order_id)
                _kds_mark_printed(order_id, [it["id"] for it in items], full=True)
            elif new_item_ids:
                print_ticket_via_component(order_id, only_items=list(new_item_ids))
                _kds_mark_printed(order_id, list(new_item_ids))

        # ======================================================
        # ITEMS GRID
        # ======================================================
        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j >= len(items):
                    continue

                item = items[i + j]
                item_id = item["id"]
                current_status = (item["kds_status"] or "pending").lower()

                css_key = current_status if current_status in (
                    "in_progress", "ready", "done"
                ) else "in_progress"

                note_text = f" — {item['note']}" if item.get("note") else ""

                with col:
                    st.markdown(
                        f"<div class='status-{css_key}'>"
                        f"{item['name']} x{item['qty']}{note_text}<br>"
                        f"<b>Status:</b> {current_status.upper()}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # ✅ FIXED DONE (UPSERT) — does NOT break print flow
                    if st.button("✅ DONE", key=f"done_item_{item_id}") and current_status != "done":
                        exists = fetchone(
                            "SELECT id FROM kds_items_status WHERE order_item_id=?",
                            (item_id,),
                        )

                        if exists:
                            execute(
                                """
                                UPDATE kds_items_status
                                SET status='done', updated_at=datetime('now')
                                WHERE order_item_id=?
                                """,
                                (item_id,),
                            )
                        else:
                            execute(
                                """
                                INSERT INTO kds_items_status (id, order_item_id, status, updated_at)
                                VALUES (lower(hex(randomblob(16))), ?, 'done', datetime('now'))
                                """,
                                (item_id,),
                            )

                        push_realtime_event("kds_updated", {"order_id": order_id})
                        st.rerun()

        st.markdown("---")

    # ======================================================
    # AUTO-PRINT EXECUTION (RESTORED)
    # ======================================================
    for (order_id, item_ids, full) in auto_print_queue:
        print_ticket_via_component(order_id, only_items=None if full else item_ids)
        _kds_mark_printed(order_id, item_ids, full=full)

    # ======================================================
    # SOUND ALERT
    # ======================================================
    if any_new_items and enable_sound:
        st.markdown(
            "<audio autoplay>"
            "<source src='https://actions.google.com/sounds/v1/alarms/beep_short.ogg' type='audio/ogg'>"
            "</audio>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

def wrap_print_html_for_popup(html_content: str) -> str:
    """Wrap minimal HTML so Chrome opens print dialog reliably."""
    return f"""
    <html>
    <head><meta charset='UTF-8'></head>
    <body onload="window.print();">
        {html_content}
    </body>
    </html>
    """

# ---------------- TOUCH CART HELPERS ----------------

def render_touch_category_bar(menu_items, context: str) -> str:
    cats = sorted({(m.get("category") or "Other") for m in menu_items}) or ["All"]
    state_key = f"active_cat_{context}"
    if state_key not in st.session_state:
        st.session_state[state_key] = cats[0]
    active = st.session_state[state_key]

    st.markdown('<div class="cat-btn-wrap">', unsafe_allow_html=True)
    for i in range(0, len(cats), 4):
        cols = st.columns(4)
        row = cats[i : i + 4]
        for j, cat in enumerate(row):
            with cols[j]:
                if st.button(cat, key=f"cat_{context}_{cat}", width="stretch"):
                    st.session_state[state_key] = cat
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ✅ GREEN CATEGORY STYLE (UI ONLY)
    st.markdown(
        """
        <style>
        .cat-btn-wrap button {
            background-color: #16a34a !important;   /* GREEN */
            color: white !important;
            border-radius: 10px !important;
            border: 2px solid #15803d !important;
            font-weight: 700 !important;
            padding: 12px !important;
        }

        .cat-btn-wrap button:hover {
            background-color: #22c55e !important;
            border-color: #16a34a !important;
            transform: scale(1.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    return active

def render_touch_item_grid(order_id, menu_items, context):
    # ------------------------------------------------------------
    # 🌍 DISPLAY CONTEXT (SAFE – NO LOGIC CHANGE)
    # ------------------------------------------------------------
    symbol = st.session_state.get("currency_symbol", "£")

    active_cat = st.session_state.get(f"active_cat_{context}", "Other")
    filtered = [m for m in menu_items if (m.get("category") or "Other") == active_cat]

    if not filtered:
        st.info("No items in this category")
        return

    st.markdown('<div class="item-grid">', unsafe_allow_html=True)

    # 4 items per row (touch-friendly)
    for i in range(0, len(filtered), 4):
        cols = st.columns(4)
        row = filtered[i:i + 4]

        for j, it in enumerate(row):
            with cols[j]:
                # ✅ FIX: dynamic currency symbol (DISPLAY ONLY)
                label = f"{it['name']}\n{symbol}{it['price']:.2f}"

                if st.button(
                    label,
                    key=f"item_{context}_{order_id}_{it['id']}",
                    width="stretch",   # ✅ FIX (2026-safe)
                ):
                    # =====================================================
                    # ✅ CREATE TAKEAWAY ORDER ON FIRST ITEM ONLY
                    # =====================================================
                    if context == "tky" and not order_id:
                        order_id = create_order(
                            "takeaway",
                            table_no=None,
                            customer_name=None,
                            paid=False,
                        )
                        st.session_state["open_takeaway_order_id"] = order_id

                    # POS-grade safety check
                    if not order_id:
                        st.error("Order not initialised")
                        return

                    add_order_item(order_id, it, 1, "")
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # 🔒 UNIFORM ITEM BUTTON SIZE (POS SAFE)
    st.markdown("""
    <style>
    .item-grid button {
        background: linear-gradient(145deg, #fff7ed, #ffedd5) !important;
        color: #111 !important;
        border-radius: 16px !important;
        border: 2px solid #fb923c !important;

        height: 110px !important;
        min-height: 110px !important;
        max-height: 110px !important;

        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;

        text-align: center !important;
        white-space: pre-line !important;
        line-height: 1.2 !important;

        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 10px !important;

        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        transition: all 0.2s ease;
    }

    .item-grid button > div {
        width: 100% !important;
        text-align: center !important;
    }

    .item-grid button:hover {
        background: linear-gradient(145deg, #fb923c, #f97316) !important;
        color: white !important;
        transform: scale(1.05);
    }

    @media (max-width: 1024px) {
        .item-grid button {
            height: 95px !important;
            min-height: 95px !important;
            max-height: 95px !important;
            font-size: 1rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def recalc_order_total(order_id: str):
    items = fetchall(
        "SELECT qty, unit_price, tax FROM order_items WHERE order_id=?",
        (order_id,),
    )

    subtotal = 0.0
    tax_total = 0.0

    for it in items:
        qty = int(it["qty"])
        price = float(it["unit_price"])
        tax_pct = float(it["tax"] or 0.0)

        line_base = qty * price
        line_tax = line_base * (tax_pct / 100.0)

        subtotal += line_base
        tax_total += line_tax

    total = subtotal + tax_total

    execute(
        """
        UPDATE orders
        SET subtotal = ?,
            tax = ?,          -- ✅ THIS WAS MISSING
            total = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            round(subtotal, 2),
            round(tax_total, 2),
            round(total, 2),
            datetime.now(timezone.utc).isoformat(),
            order_id,
        ),
    )


def render_touch_cart(order_id: str, context: str):
    order = fetchone("SELECT * FROM orders WHERE id=?", (order_id,))
    if not order:
        st.error("Order not found")
        return

    items = fetchall(
        "SELECT * FROM order_items WHERE order_id=? ORDER BY created_at",
        (order_id,),
    )

    active_item_id = st.session_state.get("active_cart_item_id")

    # ============================================================
    # 🌍 DISPLAY CONTEXT (SAFE GLOBALS)
    # ============================================================
    symbol = st.session_state.get("currency_symbol", "£")
    tax_label = st.session_state.get("tax_label", "VAT")

    # ============================================================
    # STYLES
    # ============================================================
    st.markdown("""
    <style>
    .cart-row {
        background:#f9fafb;
        border:1px solid #e5e7eb;
        border-radius:10px;
        padding:12px 14px;
        margin-bottom:6px;
        font-size:0.95rem;
        color:#111827;
    }
    .cart-row-active {
        background:#f3f4f6;
        border:2px solid #9ca3af;
    }
    .cart-header {
        font-size:1.1rem;
        font-weight:800;
        margin-bottom:10px;
    }
    .cart-total {
        font-size:1.25rem;
        font-weight:900;
        margin-top:10px;
    }
    .kds-badge {
        font-size:0.7rem;
        color:white;
        background:#6b7280;
        padding:3px 6px;
        border-radius:6px;
        width:fit-content;
        margin-top:2px;
        margin-bottom:6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # HEADER
    # ============================================================
    label = order.get("table_no") or order.get("token") or order_id[:8]
    status = (order.get("status") or "").upper()

    st.markdown(
        f"<div class='cart-header'>Order <b>{label}</b> "
        f"<span style='font-size:0.8rem;color:#6b7280;'>[{status}]</span></div>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # ITEMS
    # ============================================================
    if not items:
        st.info("No items yet.")
    else:
        for it in items:
            is_active = active_item_id == it["id"]
            qty = int(it["qty"])
            base = qty * it["unit_price"]
            tax_val = base * (it["tax"] / 100)
            line_total = base + tax_val

            row_class = "cart-row cart-row-active" if is_active else "cart-row"
            label_txt = f"{html.escape(it['name'])} ×{qty} {symbol}{line_total:.2f}"

            with st.container():
                col_label, col_x = st.columns([8, 1])

                with col_label:
                    if st.button(
                        label_txt,
                        key=f"row_{context}_{it['id']}",
                        width="stretch",
                    ):
                        st.session_state["active_cart_item_id"] = it["id"]
                        st.rerun()

                with col_x:
                    if st.button("✖", key=f"rm_{context}_{it['id']}"):
                        execute("DELETE FROM order_items WHERE id=?", (it["id"],))
                        recalc_order_total(order_id)
                        push_realtime_event("order_updated", {"order_id": order_id})
                        if st.session_state.get("active_cart_item_id") == it["id"]:
                            st.session_state.pop("active_cart_item_id", None)
                        st.rerun()

            st.markdown(f"<div class='{row_class}'></div>", unsafe_allow_html=True)

            kds = fetchone(
                "SELECT status FROM kds_items_status WHERE order_item_id=?",
                (it["id"],),
            )
            kds_status = (kds["status"] if kds else "pending").upper()
            st.markdown(
                f"<div class='kds-badge'>{kds_status}</div>",
                unsafe_allow_html=True,
            )

    # ============================================================
    # TOTAL
    # ============================================================
    totals_row = fetchone("SELECT total, tax, paid FROM orders WHERE id=?", (order_id,))
    order_total = float((totals_row or {}).get("total") or 0.0)
    order_tax = float((totals_row or {}).get("tax") or 0.0)
    order_paid_flag = int((totals_row or {}).get("paid") or 0)

    st.markdown(
        f"<div class='cart-total'>Total: {symbol}{order_total:.2f}</div>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # 🧾 PRINT BILL (BEFORE PAYMENT) — DINE-IN ONLY
    # ============================================================
    if context == "dine" and order_total > 0:
        if st.button(
            "🖨️ Print / Show Bill",
            key=f"print_bill_{order_id}",
            width="stretch",
        ):
            st.session_state["__bill_to_show__"] = order_id

    # ============================================================
    # 🍽️ DINE-IN PAYMENT + CLOSE TABLE
    # ============================================================
    if context == "dine" and order_total > 0:
        st.markdown("### 💳 Payment")

        method = st.radio(
            "Method",
            ["Cash", "Card"],
            horizontal=True,
            key=f"dine_pay_method_{order_id}",
        )

        if st.button(
            "✅ Confirm Payment",
            key=f"dine_confirm_{order_id}",
            width="stretch",
        ):
            already_paid = fetchone(
                "SELECT paid FROM orders WHERE id=?", (order_id,)
            )["paid"]

            if already_paid:
                st.warning("⚠️ Order already paid.")
            else:
                normalized_method = "Cash" if "cash" in method.lower() else "Card"

                # ✅ Store tax_amount into payments (new column)
                execute(
                    """
                    INSERT INTO payments (id, order_id, method, amount, tax_amount, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        order_id,
                        normalized_method,
                        order_total,
                        order_tax,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

                execute(
                    "UPDATE orders SET paid=1, updated_at=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), order_id),
                )

                st.success("💰 Payment recorded.")
                st.rerun()

        # ✅ CLOSE TABLE ONLY IF PAID
        if st.button(
            "🧾 Close Table",
            key=f"dine_close_{order_id}",
            width="stretch",
        ):
            # Re-check from DB to be 100% sure
            paid_chk = fetchone(
                "SELECT paid FROM orders WHERE id=?", (order_id,)
            )["paid"]

            if not paid_chk:
                st.error("❌ Cannot close table before payment. Please confirm payment first.")
            else:
                execute(
                    "UPDATE orders SET status='closed', updated_at=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), order_id),
                )
                st.session_state.pop("open_table_order_id", None)
                st.session_state.pop("active_cart_item_id", None)
                st.success("✅ Table closed.")
                st.rerun()

    # ============================================================
    # 🥡 TAKEAWAY PAYMENT
    # ============================================================
    if context == "tky" and order_total > 0:
        st.markdown("### 💳 Payment")

        method = st.radio(
            "Method",
            ["Cash", "Card"],
            horizontal=True,
            key=f"tky_pay_method_{order_id}",
        )

        if st.button(
            "✅ Confirm Payment",
            key=f"tky_confirm_{order_id}",
            width="stretch",
        ):
            already_paid = fetchone(
                "SELECT paid FROM orders WHERE id=?", (order_id,)
            )["paid"]

            if already_paid:
                st.warning("⚠️ Order already paid.")
                return

            normalized_method = "Cash" if "cash" in method.lower() else "Card"

            # ✅ Store tax_amount into payments (new column)
            execute(
                """
                INSERT INTO payments (id, order_id, method, amount, tax_amount, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    order_id,
                    normalized_method,
                    order_total,
                    order_tax,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            execute(
                """
                UPDATE orders
                SET paid=1,
                    status = CASE
                        WHEN order_type='takeaway' AND status='pending'
                        THEN 'in_progress'
                        ELSE status
                    END,
                    updated_at=?
                WHERE id=?
                """,
                (datetime.now(timezone.utc).isoformat(), order_id),
            )

            st.session_state["__bill_to_show__"] = order_id
            st.session_state.pop("open_takeaway_order_id", None)
            st.session_state.pop("active_cart_item_id", None)
            st.success("💰 Payment recorded.")
            st.rerun()

# ---------------- QR CUSTOMER ORDERING (TABLE SCAN) ----------------

def _qr_cart_key(table_no: str) -> str:
    """Session key for QR cart per table."""
    return f"qr_cart_{table_no}"

def _get_qr_cart(table_no: str) -> Dict[str, dict]:
    key = _qr_cart_key(table_no)
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]

def _qr_clear_cart(table_no: str):
    key = _qr_cart_key(table_no)
    if key in st.session_state:
        del st.session_state[key]

def _qr_add_to_cart(table_no: str, menu_row: dict):
    cart = _get_qr_cart(table_no)
    mid = menu_row["id"]
    if mid in cart:
        cart[mid]["qty"] += 1
    else:
        cart[mid] = {
            "menu_id": mid,
            "name": menu_row["name"],
            "price": float(menu_row["price"]),
            "tax": float(menu_row["tax"] or 0.0),
            "qty": 1,
            "note": "",
        }

def _qr_update_qty(table_no: str, menu_id: str, delta: int):
    cart = _get_qr_cart(table_no)
    if menu_id in cart:
        cart[menu_id]["qty"] += delta
        if cart[menu_id]["qty"] <= 0:
            del cart[menu_id]

def _qr_set_note(table_no: str, menu_id: str, note: str):
    cart = _get_qr_cart(table_no)
    if menu_id in cart:
        cart[menu_id]["note"] = note.strip()

def _render_qr_item_grid(table_no: str, menu_items):
    """Customer-facing menu grid: tap to add to QR cart (does NOT directly hit DB)."""
    active_cat = st.session_state.get("active_cat_qr", "Other")
    filtered = [m for m in menu_items if (m.get("category") or "Other") == active_cat]

    st.markdown('<div class="item-btn-wrap">', unsafe_allow_html=True)
    for it in filtered:
        label = f"{it['name']} (£{it['price']:.2f})"
        if st.button(
            label,
            key=f"qr_item_{table_no}_{it['id']}",
            width="stretch",
        ):
            _qr_add_to_cart(table_no, it)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def _render_qr_cart(table_no: str):
    cart = _get_qr_cart(table_no)
    if not cart:
        st.info("No items in your order yet. Tap items on the left to add them.")
        return

    st.markdown("### 🧺 Your Order")
    total = 0.0

    for mid, entry in list(cart.items()):
        qty = entry["qty"]
        price = entry["price"]
        tax = entry["tax"]
        base = qty * price
        tax_val = base * (tax / 100.0)
        line_total = base + tax_val
        total += line_total

        name_safe = html.escape(entry["name"])

        c1, c2, c3, c4 = st.columns([4, 1, 1, 2])
        with c1:
            st.markdown(f"**{name_safe}**", unsafe_allow_html=True)
        with c2:
            if st.button("➖", key=f"qr_minus_{mid}"):
                _qr_update_qty(table_no, mid, -1)
                st.rerun()
        with c3:
            st.markdown(f"x{qty}")
        with c4:
            st.markdown(f"£{line_total:.2f}")

        # Optional note per item
        note_key = f"qr_note_{table_no}_{mid}"
        current_note = entry.get("note", "")
        new_note = st.text_input(
            "Note (no onions, extra cheese, etc.)",
            value=current_note,
            key=note_key,
        )
        if new_note != current_note:
            _qr_set_note(table_no, mid, new_note)

        st.markdown("---")

    st.markdown(f"### Total (estimate): **£{total:.2f}**")
    st.caption("Final bill will be generated by the restaurant based on these items.")

# ---------------- QR CUSTOMER ORDERING (TABLE SCAN) ----------------

QR_STYLE = """
<style>
body, .main {
  background: #ffffff !important;
  color: #111 !important;
  font-family: 'Inter', sans-serif !important;
}
.qr-header {
  text-align: center;
  margin-top: 10px;
  margin-bottom: 10px;
}
.qr-header h2 {
  color: #ff9500;
  font-weight: 700;
  margin-bottom: 6px;
}
.qr-item-card {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.qr-btn {
  background: #ff9500 !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: 10px !important;
  width: 100% !important;
}
.qr-btn:disabled {
  opacity: 0.5;
}
.qr-cart {
  background: #fff8f0;
  border: 1px solid #ffe1b3;
  border-radius: 12px;
  padding: 12px;
  margin-top: 10px;
}
.qr-cart-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px dashed #fcd34d;
  font-size: 0.9rem;
}
.qr-total {
  font-size: 1.05rem;
  font-weight: 700;
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
}
@media(max-width:768px){
  .qr-container { flex-direction: column !important; }
  .qr-left, .qr-right { width: 100% !important; margin: 0 !important; }
  .sticky-submit {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #ff9500; color: #fff;
    text-align: center; padding: 14px;
    font-weight: 700;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
  }
}
</style>
"""

def _qr_key(table_no): return f"qr_cart_{table_no.upper()}"

def _get_cart(table_no):
    key=_qr_key(table_no)
    if key not in st.session_state: st.session_state[key]={}
    return st.session_state[key]

def _save_cart(table_no, cart): st.session_state[_qr_key(table_no)] = cart
def _clear_cart(table_no): st.session_state.pop(_qr_key(table_no), None)

def qr_order_page(table_no: str = None):
    """
    Customer QR Ordering Page
    • ONE active order per table (QR or DINE)
    • Add-ons always append to SAME order
    • Approval remains mandatory
    • Customer can cancel items BEFORE placing
    • Customer can ADD MORE even after placing (pending approval)
    """
    import html, time, uuid
    from streamlit_autorefresh import st_autorefresh

    # ---------------- AUTO REFRESH ----------------
    try:
        st_autorefresh(interval=5000, key=f"qr_autorefresh_{table_no}")
    except Exception:
        time.sleep(5)
        st.rerun()

    # ---------------- PARAMS ----------------
    try:
        params = st.query_params
    except Exception:
        params = st.experimental_get_query_params()

    if not table_no:
        table_no = params.get("table")
    sig = params.get("sig")

    if isinstance(table_no, list):
        table_no = table_no[0] if table_no else None
    if isinstance(sig, list):
        sig = sig[0] if sig else None

    if not table_no:
        st.error("Invalid QR link — no table specified.")
        return

    table_no = str(table_no).upper().strip()

    # ============================================================
    # ✅ SAFE QR SIGNATURE VALIDATION (WON'T BLOCK PAGE)
    # ============================================================
    try:
        validate_qr_sig(table_no, sig, strict=False)
    except Exception:
        # Do NOT block customer ordering page
        st.warning("⚠️ QR security signature missing/invalid. Demo mode access enabled.")

    cart_key = f"qr_cart_{table_no}"

    # ============================================================
    # BUSINESS PROFILE
    # ============================================================
    profile = get_business_profile()
    currency = (profile.get("currency") or "GBP").upper()

    CURRENCY_SYMBOLS = {
        "GBP": "£", "INR": "₹", "AED": "د.إ",
        "QAR": "ر.ق", "SAR": "﷼", "OMR": "﷼",
    }
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")

    tax_label = "GST" if currency == "INR" else "VAT"

    # ============================================================
    # HEADER
    # ============================================================
    st.markdown(
        f"""
        <div style='text-align:center'>
            <h2>{html.escape(profile.get("business_name") or "My Business")}</h2>
            <div>Table <b>{table_no}</b></div>
            <p style='font-size:0.8rem'>For Special Requirements call waiter</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # ACTIVE ORDER
    # ============================================================
    active_order = get_active_table_order(table_no)

    existing_items = []
    if active_order:
        existing_items = fetchall(
            """
            SELECT name, qty, unit_price, tax
            FROM order_items
            WHERE order_id=?
            ORDER BY created_at
            """,
            (active_order["id"],),
        )

    # ============================================================
    # MENU
    # ============================================================
    menu = fetchall("SELECT * FROM menu WHERE available=1 ORDER BY category, name")
    if not menu:
        st.warning("Menu unavailable.")
        return

    cart = st.session_state.get(cart_key, {})

    # ============================================================
    # CATEGORY BAR
    # ============================================================
    categories = sorted({m.get("category") or "Other" for m in menu})
    cat_key = f"qr_cat_{table_no}"

    cols = st.columns(min(4, len(categories)))
    for i, c in enumerate(categories):
        with cols[i % len(cols)]:
            if st.button(c, key=f"cat_{table_no}_{c}", width="stretch"):
                st.session_state[cat_key] = c
                st.rerun()

    active_cat = st.session_state.get(cat_key)

    left, right = st.columns(2)

    # ============================================================
    # MENU SIDE
    # ============================================================
    with left:
        if not active_cat:
            st.info("Select a category")
        else:
            for it in [m for m in menu if (m.get("category") or "Other") == active_cat]:
                if st.button(
                    f"{html.escape(it['name'])} — {symbol}{it['price']:.2f}",
                    key=f"tap_{table_no}_{it['id']}",
                    width="stretch",
                ):
                    cart.setdefault(it["id"], {"menu": it, "qty": 0})
                    cart[it["id"]]["qty"] += 1
                    st.session_state[cart_key] = cart
                    st.rerun()

    # ============================================================
    # CART SIDE
    # ============================================================
    with right:
        st.markdown("### 🛒 Your Cart")

        if existing_items:
            st.markdown("#### ✅ Already Ordered")
            for it in existing_items:
                st.markdown(f"{it['name']} ×{it['qty']}")

        if cart:
            st.markdown("#### ➕ New Items")
            for k, entry in list(cart.items()):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"{entry['menu']['name']} ×{entry['qty']}")
                with c2:
                    if st.button("❌", key=f"rm_{table_no}_{k}"):
                        cart.pop(k)
                        st.session_state[cart_key] = cart
                        st.rerun()

        # ========================================================
        # PLACE ORDER (SAFE APPEND)
        # ========================================================
        if cart:
            if st.button("🚀 Place Order", width="stretch"):
                if active_order and int(active_order.get("paid") or 0) == 0:
                    order_id = active_order["id"]
                else:
                    order_id = str(uuid.uuid4())
                    execute(
                        """
                        INSERT INTO orders
                        (id, order_type, table_no, status, approved, created_at, updated_at)
                        VALUES (?, 'qr', ?, 'pending', 0, datetime('now'), datetime('now'))
                        """,
                        (order_id, table_no),
                    )

                for entry in cart.values():
                    add_order_item(order_id, entry["menu"], entry["qty"], "")

                st.session_state[cart_key] = {}
                push_realtime_event("order_updated", {"order_id": order_id})
                st.success("✅ Order sent for approval. You may add more items.")
                st.rerun()

        # ========================================================
        # REQUEST BILL
        # ========================================================
        if st.button(
            "💰 Request Bill",
            width="stretch",
            disabled=(len(existing_items) + len(cart)) == 0,
        ):
            push_realtime_event("bill_request", {"table_no": table_no})
            st.rerun()




# ---------------- HOME DASHBOARD (BIG TILES) ----------------

HOME_MENU_CSS = """
<style>
#home-menu-wrapper h2 {
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}
#home-menu-wrapper .stButton>button {
    width: 100%;
    height: 170px;
    border-radius: 20px;
    font-size: 1.2rem;
    font-weight: 600;
    padding: 0;
    border: 1px solid #1f2937;
}
#home-menu-wrapper .stButton>button:hover {
    border: 2px solid #14b8a6;
}
</style>
"""

def home_dashboard():
    st.markdown("""
    <style>

    /* Fullscreen center alignment */
    .brand-wrapper {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;

        height: 85vh;
        width: 100%;
        text-align: center;
    }

    /* MASSIVE product title */
    .brand-title {
        font-size: 140px;
        font-weight: 900;
        font-family: 'Inter', sans-serif;
        letter-spacing: -3px;
        color: #1F4ED8;
        text-shadow: 0px 0px 40px rgba(20, 184, 166, 0.35);
        margin-bottom: 20px;
    }

    /* Product subtitle */
    .brand-sub {
        font-size: 24px;
        font-weight: 400;
        color: #1F4ED8;
        margin-top: -20px;
    }

    </style>

    <div class="brand-wrapper">
        <div class="brand-title">Que-POS</div>
        <div class="brand-sub">
            Smart Restaurant Billing & Kitchen Display System
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN TOUCH ORDER BUILDER ----------------

TOUCH_POS_CSS = """
<style>
.touch-root {
    display: flex;
    flex-direction: column;
    height: auto !important;
    padding: 0 !important;
    margin: 0 !important;
}
.touch-cart {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-height: 70vh;
    border-radius: 16px;
    background: #111827;
    color: white;
    padding: 12px;
    margin-left: 12px;
}
.touch-cart-header {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.touch-cart-items {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 8px;
}
.touch-cart-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    font-size: 0.85rem;
    padding: 4px 0;
    border-bottom: 1px dashed rgba(255,255,255,0.08);
}
.touch-cart-row-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.touch-cart-row-qty {
    flex: 0 0 auto;
    min-width: 40px;
    text-align: right;
}
.touch-cart-row-total {
    flex: 0 0 auto;
    min-width: 70px;
    text-align: right;
}
.touch-cart-totals {
    border-top: 1px solid rgba(255,255,255,0.2);
    padding-top: 6px;
    margin-bottom: 6px;
    font-size: 0.95rem;
}
.touch-cart-actions {
    margin-top: 4px;
}
.touch-cart-actions button {
    width: 100%;
}
</style>
"""
RESPONSIVE_MOBILE_FIXES = """
<style>
/* ---------------------------------------------
   📱 MOBILE + TABLET RESPONSIVE FIXES (Que-POS)
----------------------------------------------*/
@media (max-width: 992px) {
    /* Reduce container padding for small screens */
    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        padding-top: 0.5rem !important;
    }

    /* Cards and sections */
    .section-card {
        margin: 0.3rem 0 !important;
        padding: 0.9rem !important;
        border-radius: 12px !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.08);
        background: #ffffff !important;
        color: #333 !important;
    }

    /* Buttons become full width */
    button, .stButton>button {
        width: 100% !important;
        font-size: 15px !important;
        padding: 12px !important;
        border-radius: 10px !important;
        background-color: #ff7f32 !important;
        color: white !important;
    }

    /* Reduce font sizes slightly */
    h1, h2, h3, h4 {
        font-size: 1.1rem !important;
    }
    p, label, span, div, input, select {
        font-size: 0.95rem !important;
    }

    /* Tables & dataframes */
    [data-testid="stDataFrame"], .stTable {
        font-size: 12px !important;
        overflow-x: auto !important;
    }

    /* Fix sidebar scroll on mobile */
    [data-testid="stSidebar"] {
        width: 85vw !important;
        overflow-y: auto !important;
    }
}
</style>
"""

# ==========================================================
#   WHITE + ORANGE CUSTOMER QR ORDER PAGE (FINAL – NEW API)
# ==========================================================
import html, streamlit as st
from datetime import datetime

# ---------------- THEMES & GLOBAL CSS ----------------

THEME_ENGLISH_DARK = """
<style>
body, .main {
    background: #111827; /* Deep dark background */
    color: #f9fafb;
    font-family: 'Inter', sans-serif;
}

/* ---------- SECTION CARDS ---------- */
.section-card {
    background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(251, 146, 60, 0.3);
    box-shadow: 0 4px 20px rgba(251, 146, 60, 0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.section-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(251, 146, 60, 0.3);
}

/* ---------- HEADINGS ---------- */
h1, h2, h3, h4, h5, h6 {
    color: #fbbf24;
    font-weight: 700;
    letter-spacing: 0.5px;
}
h3 {
    background: linear-gradient(90deg, #fb923c, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: linear-gradient(135deg, #fb923c 0%, #f97316 100%);
    color: white !important;
    border-radius: 30px;
    border: none;
    padding: 14px 22px;
    font-size: 40px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 12px rgba(251, 146, 60, 0.4);
}
.stButton>button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #f97316 0%, #fb923c 100%);
    box-shadow: 0 6px 18px rgba(251, 146, 60, 0.6);
}

/* ---------- INPUTS & SELECTBOX ---------- */
input, textarea, select, .stSelectbox, .stTextInput, .stCheckbox {
    color: #f9fafb !important;
}
.stTextInput>div>div>input, .stSelectbox>div>div>select {
    background-color: #1f2937 !important;
    border: 1px solid #fb923c33 !important;
    border-radius: 8px;
    padding: 8px 10px;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
    border-color: #fb923c !important;
    box-shadow: 0 0 8px rgba(251, 146, 60, 0.4);
}

/* ---------- METRIC & TAG COLORS ---------- */
[data-testid="stMetricValue"] {
    color: #fb923c !important;
}
[data-testid="stMetricDelta"] {
    color: #f59e0b !important;
}

/* ---------- SCROLLBARS ---------- */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #fb923c, #f97316);
    border-radius: 6px;
}
::-webkit-scrollbar-track {
    background: #1f2937;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1f2937;
    color: #f9fafb;
    border-radius: 8px;
    padding: 10px 16px;
    transition: all 0.2s ease;
    border: 1px solid #374151;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #fb923c, #f97316);
    color: #fff !important;
    font-weight: 700;
    box-shadow: 0 3px 10px rgba(251, 146, 60, 0.3);
}

/* ---------- MISC ELEMENTS ---------- */
hr {
    border: none;
    border-top: 1px solid #374151;
    margin: 18px 0;
}
</style>
"""


THEME_SIDEBAR_FM_MINT = """
<style>
[data-testid="stSidebar"] {
    background: #4169E1	 !important;
    color: #2d3e40 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #2d3e40 !important;
}
[data-testid="stSidebar"] .stButton>button {
    background: white !important;
    color: #2d3e40 !important;
    border: 1px solid #2d3e40 !important;
    border-radius: 8px;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background: #2d3e40 !important;
    color: white !important;
}
</style>
"""

SIDEBAR_LICENSE_TEXT_FIX = """
<style>
[data-testid="stSidebar"] .license-text,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #000000 !important;
}
</style>
"""

SIDEBAR_SHOW_TOGGLE = """
<style>
button[title="Hide sidebar"],
button[title="Show sidebar"] {
    display: flex !important;
    visibility: visible ! important;
    opacity: 1 ! important;
}
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
</style>
"""

FIX_ALL_GAPS = """
<style>
html, body {
    margin: 0 !important;
    padding: 0 !important;
}
header[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}
main[data-testid="stMain"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
div[data-testid="stVerticalBlock"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
"""
RESPONSIVE_MOBILE_FIXES = """
<style>
/* ===========================================================
   📱 Que-POS RESPONSIVE DESIGN (Phones + Tablets)
   Author: Arjvan IT | Updated for Orange-White Theme
   =========================================================== */

/* Base color variables for consistency */
:root {
  --qpos-orange: #ff7f32;
  --qpos-dark: #2d2d2d;
  --qpos-bg: #ffffff;
  --qpos-lightgray: #f7f7f7;
}

/* ----------- Mobile (max 767px) ------------ */
@media (max-width: 767px) {
  .block-container {
    padding: 0.8rem !important;
  }

  .section-card {
    padding: 0.9rem !important;
    border-radius: 12px !important;
    background: var(--qpos-bg) !important;
    color: #333 !important;
    box-shadow: 0 3px 6px rgba(0,0,0,0.08);
  }

  button, .stButton>button {
    width: 100% !important;
    background-color: var(--qpos-orange) !important;
    color: white !important;
    font-size: 15px !important;
    padding: 12px !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
  }

  h1, h2, h3, h4 {
    font-size: 1.1rem !important;
  }

  input, select, textarea, label, p, span, div {
    font-size: 0.95rem !important;
  }

  /* Login and forms stack vertically */
  form, .stForm {
    display: block !important;
  }

  /* Sidebar width for small screens */
  [data-testid="stSidebar"] {
    width: 80vw !important;
    overflow-y: auto !important;
  }

  /* Compact data tables */
  [data-testid="stDataFrame"], .stTable {
    font-size: 12px !important;
    overflow-x: auto !important;
  }
}

/* ----------- Tablet (768px – 1024px) ------------ */
@media (min-width: 768px) and (max-width: 1024px) {
  /* Tablet layout (iPad / Android tabs) */
  .block-container {
    padding: 1.2rem !important;
  }

  /* ✅ Two-column layout for forms, login, or order sections */
  .qpos-two-col, .stForm .qpos-two-col, form.qpos-two-col {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem !important;
    align-items: start !important;
  }

  /* Optional: ensures inputs align nicely on tablets */
  .qpos-two-col .stTextInput, 
  .qpos-two-col .stSelectbox, 
  .qpos-two-col .stNumberInput {
    width: 100% !important;
  }

  .section-card {
    border-radius: 14px !important;
    background: var(--qpos-bg) !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.1);
  }

  button, .stButton>button {
    width: 100% !important;
    padding: 14px !important;
    background-color: var(--qpos-orange) !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
  }

  [data-testid="stSidebar"] {
    width: 320px !important;
  }

  h1, h2 {
    font-size: 1.4rem !important;
  }

  /* Table readability on tablets */
  [data-testid="stDataFrame"], .stTable {
    font-size: 13px !important;
    overflow-x: auto !important;
  }
}
</style>
"""

def main():
    st.set_page_config(
        page_title="Que-POS",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ============================================================
    # 🔓 PUBLIC QR ACCESS (MUST RUN FIRST – NO LICENSE / LOGIN)
    # ============================================================
    try:
        params = st.query_params
    except Exception:
        params = st.experimental_get_query_params()

    qr_val = params.get("qr")
    table_val = params.get("table")

    if isinstance(qr_val, list):
        qr_val = qr_val[0]
    if isinstance(table_val, list):
        table_val = table_val[0]

    if qr_val == "1" and table_val:
        qr_order_page(table_val)
        st.stop()   # ⛔ stop everything else (NO license, NO login)

    # ============================================================
    # 🔷 GLOBAL NEON BRANDING (ALL PAGES – SAFE)
    # ============================================================
    try:
        profile = get_business_profile()
    except Exception:
        profile = {}

    brand_name = html.escape(profile.get("business_name") or "Que-POS")

    st.markdown(f"""
    <style>
    h1, h2, h3, h4 {{
        color: #1F4ED8 !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
    }}
    .nav-col h1, .nav-col h2, .nav-col h3 {{
        color: #1F4ED8 !important;
    }}
    .qpos-brand-global {{
        position: fixed;
        top: 14px;
        right: 24px;
        z-index: 9999;
        font-family: Inter, sans-serif;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: -1px;
        color: #1F4ED8;
        user-select: none;
        text-shadow:
            0 0 6px rgba(0,229,255,0.9),
            0 0 14px rgba(0,229,255,0.8),
            0 0 28px rgba(0,229,255,0.6);
    }}
    </style>
    <div class="qpos-brand-global">{brand_name}</div>
    """, unsafe_allow_html=True)

    # ============================================================
    # INIT SYSTEMS
    # ============================================================
    if not getattr(sys, "frozen", False):
        if "ws_started" not in st.session_state:
            start_ws_listener()
            st.session_state["ws_started"] = True

    init_db()

    # ============================================================
    # AUTO CLEANUP
    # ============================================================
    try:
        now = time.time()
        last = st.session_state.get("_last_cleanup_time", 0)
        if now - last > 600:
            cleanup_rejected_orders()
            st.session_state["_last_cleanup_time"] = now
    except Exception:
        pass

    # ============================================================
    # GLOBAL THEMES / CSS
    # ============================================================
    st.markdown(THEME_ENGLISH_DARK, unsafe_allow_html=True)
    st.markdown(TOUCH_POS_CSS, unsafe_allow_html=True)
    st.markdown(FIX_ALL_GAPS, unsafe_allow_html=True)

    try:
        st.markdown(THEME_SIDEBAR_FM_MINT, unsafe_allow_html=True)
        st.markdown(SIDEBAR_LICENSE_TEXT_FIX, unsafe_allow_html=True)
        st.markdown(SIDEBAR_SHOW_TOGGLE, unsafe_allow_html=True)
        st.markdown(RESPONSIVE_MOBILE_FIXES, unsafe_allow_html=True)
    except NameError:
        pass

    # ============================================================
    # LICENSE CHECK
    # ============================================================
    active_license = get_active_license()
    if not active_license:
        license_activation_screen()
        return

    ok, msg = check_license_and_date()
    if not ok:
        st.error(msg)
        return

    show_expiry_warning_if_needed(active_license)

    # ============================================================
    # LOGIN (NO DOUBLE FORM)
    # ============================================================
    if "user" not in st.session_state:
        with st.form("login", clear_on_submit=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if login(username, password):
                    st.session_state["active_page"] = "Home"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return

    user = st.session_state["user"]
    role = user["role"]

    # ============================================================
    # ROLE BASED MODULES
    # ============================================================
    all_modules = [
        "Home", "Orders", "KDS", "Takeaway Status",
        "Inventory", "Menu", "Payments", "Reports", "Admin"
    ]

    try:
        allowed = user.get("allowed_modules", [])
        if isinstance(allowed, str):
            allowed = json.loads(allowed)
    except Exception:
        allowed = []

    if not allowed:
        if role == "superadmin":
            allowed = all_modules.copy()
        elif role == "manager":
            allowed = ["Home", "Orders", "KDS", "Payments", "Reports", "Inventory", "Menu"]
        elif role == "cashier":
            allowed = ["Home", "Orders", "Payments", "Reports"]
        elif role == "kitchen":
            allowed = ["Home", "KDS"]
        else:
            allowed = ["Home"]

    visible_modules = [m for m in all_modules if m in allowed]
    if not visible_modules:
        visible_modules = ["Home"]

    st.session_state.setdefault("active_page", "Home")

    # ============================================================
    # FIXED POS LAYOUT
    # ============================================================
    nav_col, main_col = st.columns([1.2, 4.8])

    with nav_col:
        st.markdown('<div class="nav-col">', unsafe_allow_html=True)
        st.markdown(f"**{user['username']}** ({role})")

        for mod in visible_modules:
            is_active = st.session_state["active_page"] == mod
            btn_type = "primary" if is_active else "secondary"
            if st.button(mod, type=btn_type, width="stretch"):
                st.session_state["active_page"] = mod
                st.rerun()

        st.markdown("---")
        st.markdown(f"🧾 **{active_license['customer_name']}**")
        st.markdown(f"⏳ Valid till: {active_license['expires_on']}")

        if st.button("🚪 Sign out", width="stretch"):
            logout()
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with main_col:
        page = st.session_state["active_page"]

        if page == "Home":
            home_dashboard()
        elif page == "Orders":
            order_builder()
        elif page == "KDS":
            kds_view()
        elif page == "Takeaway Status":
            takeaway_status_panel()
        elif page == "Inventory":
            inventory_panel()
        elif page == "Menu":
            menu_management()
        elif page == "Payments":
            payments_panel()
        elif page == "Reports":
            reports_panel()
        elif page == "Admin":
            admin_panel()



if __name__ == "__main__":
    main()









