import sqlite3
import threading
import os
from typing import List, Dict, Any

# You can override the DB path via environment variable if needed
DB_PATH = os.environ.get("DB_PATH", "sentinel_vision.db")

# Global lock to ensure explicit serialization for concurrent SQLite writes
# from multiple parallel FastAPI or vision pipeline threads.
db_lock = threading.Lock()

def _get_connection() -> sqlite3.Connection:
    """
    Returns a SQLite connection configured for multithreaded use.
    check_same_thread=False is required for sharing connections/logic across threads.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    # Enable Write-Ahead Logging (WAL) for better concurrency and performance
    conn.execute("PRAGMA journal_mode=WAL;")
    
    # Return rows as dictionary-like objects
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    Initialize the database, creating the necessary tables if they do not exist.
    """
    with db_lock:
        conn = _get_connection()
        try:
            cursor = conn.cursor()
            
            # 1. violations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    registration_string TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT NOT NULL,
                    status TEXT DEFAULT 'Pending'
                )
            ''')
            
            # 2. accidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity_level INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    lat_long_mock TEXT NOT NULL
                )
            ''')
            
            # 3. traffic_signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    junction_name TEXT NOT NULL,
                    current_density REAL NOT NULL,
                    allocated_green_time INTEGER NOT NULL
                )
            ''')
            
            conn.commit()
        finally:
            conn.close()

def insert_violation(
    timestamp: str, 
    violation_type: str, 
    registration_string: str, 
    confidence: float, 
    image_path: str, 
    status: str = 'Pending'
) -> int:
    """
    Inserts a new violation record into the database in a thread-safe manner.
    Returns the generated primary key (id).
    """
    with db_lock:
        conn = _get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations (
                    timestamp, violation_type, registration_string, 
                    confidence, image_path, status
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, violation_type, registration_string, confidence, image_path, status))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

def insert_accident(
    timestamp: str, 
    severity_level: int, 
    description: str, 
    lat_long_mock: str
) -> int:
    """
    Inserts a new accident record into the database in a thread-safe manner.
    Returns the generated primary key (id).
    """
    with db_lock:
        conn = _get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO accidents (
                    timestamp, severity_level, description, lat_long_mock
                )
                VALUES (?, ?, ?, ?)
            ''', (timestamp, severity_level, description, lat_long_mock))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

def get_violations_by_plate(registration_string: str) -> List[Dict[str, Any]]:
    """
    Retrieves all violation records for a specific license plate.
    Returns a list of dictionaries representing the rows.
    """
    # Using the lock ensures strict read/write serialization, avoiding "database is locked" 
    # errors under heavy concurrent load even with WAL.
    with db_lock:
        conn = _get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM violations
                WHERE registration_string = ?
                ORDER BY timestamp DESC
            ''', (registration_string,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

def update_signal_metrics(
    junction_name: str, 
    current_density: float, 
    allocated_green_time: int
) -> int:
    """
    Updates the traffic signal metrics for a given junction.
    If the junction does not exist, it inserts a new record.
    Returns the primary key (id) of the updated or inserted row.
    """
    with db_lock:
        conn = _get_connection()
        try:
            cursor = conn.cursor()
            
            # Check if the junction already exists
            cursor.execute('''
                SELECT id FROM traffic_signals WHERE junction_name = ?
            ''', (junction_name,))
            row = cursor.fetchone()
            
            if row:
                # Update existing record
                cursor.execute('''
                    UPDATE traffic_signals
                    SET current_density = ?, allocated_green_time = ?
                    WHERE junction_name = ?
                ''', (current_density, allocated_green_time, junction_name))
                row_id = row['id']
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO traffic_signals (
                        junction_name, current_density, allocated_green_time
                    )
                    VALUES (?, ?, ?)
                ''', (junction_name, current_density, allocated_green_time))
                row_id = cursor.lastrowid
                
            conn.commit()
            return row_id
        finally:
            conn.close()
