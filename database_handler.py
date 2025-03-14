# database_handler.py
import sqlite3
import datetime
from contextlib import contextmanager
from config import CONFIG
import io
from PIL import Image
import numpy as np

class DatabaseManager:
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(CONFIG["DATABASE_NAME"])
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_institute_id TEXT UNIQUE,
                    name TEXT,
                    encoding BLOB,
                    profile_photo BLOB
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS entry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id INTEGER,
                    employee_name TEXT,
                    entry_time DATETIME,
                    FOREIGN KEY(employee_id) REFERENCES employees(id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS exit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id INTEGER,
                    employee_name TEXT,
                    exit_time DATETIME,
                    FOREIGN KEY(employee_id) REFERENCES employees(id)
                )
            ''')
            # New stranger logs tables (do not store persistent stranger data)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stranger_entry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stranger_face BLOB,
                    entry_time DATETIME
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stranger_exit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stranger_face BLOB,
                    exit_time DATETIME
                )
            ''')
            # NEW: Visitor table – stores visitor face embedding, details, purpose, allowed exit time and status.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS visitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_name TEXT,
                    face_embedding BLOB,
                    purpose_of_visit TEXT,
                    allowed_time DATETIME,
                    visitor_status TEXT
                )
            ''')
            # NEW: Visitor entry logs table – logs visitor id and entry time.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS visitor_entry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id INTEGER,
                    entry_time DATETIME,
                    FOREIGN KEY(visitor_id) REFERENCES visitors(id)
                )
            ''')
            # NEW: Visitor exit logs table – logs visitor id and exit time.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS visitor_exit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id INTEGER,
                    exit_time DATETIME,
                    FOREIGN KEY(visitor_id) REFERENCES visitors(id)
                )
            ''')

            conn.commit()
    
    def save_employee(self, employee_institute_id, name, embedding, profile_photo):
        """Save employee data to database"""
        with self.get_connection() as conn:
            try:
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                profile_photo.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                conn.execute(
                    "INSERT INTO employees (employee_institute_id, name, encoding, profile_photo) VALUES (?, ?, ?, ?)",
                    (employee_institute_id, name, embedding.tobytes(), img_byte_arr)
                )
                conn.commit()
                print(f"Employee {name} saved successfully")
            except sqlite3.IntegrityError:
                print(f"Employee with ID {employee_institute_id} already exists!")
                raise
    
    def update_employee_embedding(self, employee_institute_id, new_embedding):
        """Update the embedding for an employee"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE employees SET encoding = ? WHERE employee_institute_id = ?",
                (new_embedding.tobytes(), employee_institute_id)
            )
            conn.commit()
    
    def log_entry(self, employee_id, employee_name):
        """Record employee entry"""
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO entry_logs (employee_id, employee_name, entry_time) VALUES (?, ?, ?)",
                (employee_id, employee_name, datetime.datetime.now())
            )
            conn.commit()
    
    def log_exit(self, employee_id, employee_name):
        """Record employee exit"""
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO exit_logs (employee_id, employee_name, exit_time) VALUES (?, ?, ?)",
                (employee_id, employee_name, datetime.datetime.now())
            )
            conn.commit()
    
    def get_employee_data(self):
        """Retrieve all employee data"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT id, employee_institute_id, name, encoding FROM employees")
            employees = []
            for row in cursor:
                employees.append({
                    'id': row['id'],
                    'employee_institute_id': row['employee_institute_id'],
                    'name': row['name'],
                    'encoding': np.frombuffer(row['encoding'], dtype=np.float32)
                })
            return employees
    
    def get_employee_photo(self, employee_institute_id):
        """Retrieve employee's profile photo"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT profile_photo FROM employees WHERE employee_institute_id = ?", (employee_institute_id,))
            result = cursor.fetchone()
            if result:
                return Image.open(io.BytesIO(result['profile_photo']))
            return None
    
    def get_last_entry(self, employee_id):
        """Get the last entry log for an employee"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(entry_time) FROM entry_logs WHERE employee_id = ?", 
                (employee_id,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def get_last_exit(self, employee_id):
        """Get the last exit log for an employee"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(exit_time) FROM exit_logs WHERE employee_id = ?", 
                (employee_id,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
        
    def get_employee_id(self, employee_institute_id):
        """Get employee ID from institute ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM employees WHERE employee_institute_id = ?", (employee_institute_id,))
            result = cursor.fetchone()
            return result
        
    def get_employee_name(self, employee_institute_id):
        """Get employee ID from institute ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM employees WHERE employee_institute_id = ?", (employee_institute_id,))
            result = cursor.fetchone()
            return result
        

    def get_entry_logs(self, count):
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT employee_name, entry_time FROM entry_logs ORDER BY entry_time DESC LIMIT ?",
                (count,)
            )
            return cursor.fetchall()

    def get_exit_logs(self, count):
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT employee_name, exit_time FROM exit_logs ORDER BY exit_time DESC LIMIT ?",
                (count,)
            )
            return cursor.fetchall()

    def delete_entry_log(self, log_id):
        with self.get_connection() as conn:
            try:
                conn.execute("DELETE FROM entry_logs WHERE id = ?", (log_id,))
                conn.commit()
                return True
            except sqlite3.Error:
                return False

    def delete_exit_log(self, log_id):
        with self.get_connection() as conn:
            try:
                conn.execute("DELETE FROM exit_logs WHERE id = ?", (log_id,))
                conn.commit()
                return True
            except sqlite3.Error:
                return False
            

    def get_employee_details(self, employee_institute_id):
        with self.get_connection() as conn:
            emp = conn.execute("SELECT * FROM employees WHERE employee_institute_id = ?", (employee_institute_id,)).fetchone()
            entry_count = conn.execute("SELECT COUNT(*) FROM entry_logs WHERE employee_id = ?", (emp['id'],)).fetchone()[0]
            exit_count = conn.execute("SELECT COUNT(*) FROM exit_logs WHERE employee_id = ?", (emp['id'],)).fetchone()[0]
            
            last_entry = conn.execute("SELECT MAX(entry_time) FROM entry_logs WHERE employee_id = ?", (emp['id'],)).fetchone()[0]
            last_exit = conn.execute("SELECT MAX(exit_time) FROM exit_logs WHERE employee_id = ?", (emp['id'],)).fetchone()[0]
            
            if last_entry and (not last_exit or last_entry > last_exit):
                last_log_type = 'entry'
                last_log_time = last_entry
                current_status = 'entry'
            elif last_exit:
                last_log_type = 'exit'
                last_log_time = last_exit
                current_status = 'exit'
            else:
                last_log_type = None
                last_log_time = None
                current_status = None

            return {
                'employee_institute_id': emp['employee_institute_id'],
                'name': emp['name'],
                'photo': Image.open(io.BytesIO(emp['profile_photo'])),
                'entry_count': entry_count,
                'exit_count': exit_count,
                'last_log_type': last_log_type,
                'last_log_time': last_log_time,
                'current_status': current_status
            }

    def get_entry_logs_by_date(self, date):
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT entry_logs.id, employees.name, entry_logs.entry_time FROM entry_logs "
                "JOIN employees ON entry_logs.employee_id = employees.id "
                "WHERE DATE(entry_time) = ? ORDER BY entry_time DESC",
                (date.strftime("%Y-%m-%d"),)
            )
            return cursor.fetchall()

    def get_exit_logs_by_date(self, date):
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT exit_logs.id, employees.name, exit_logs.exit_time FROM exit_logs "
                "JOIN employees ON exit_logs.employee_id = employees.id "
                "WHERE DATE(exit_time) = ? ORDER BY exit_time DESC",
                (date.strftime("%Y-%m-%d"),)
            )
            return cursor.fetchall()
    def get_stranger_entry_logs_by_date(self, date):
        """
        Retrieve stranger entry logs for a given date.
        Returns a list of rows with columns: id, stranger_face, entry_time.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, stranger_face, entry_time FROM stranger_entry_logs WHERE DATE(entry_time) = ? ORDER BY entry_time DESC",
                (date.strftime("%Y-%m-%d"),)
            )
            return cursor.fetchall()

    def get_stranger_exit_logs_by_date(self, date):
        """
        Retrieve stranger exit logs for a given date.
        Returns a list of rows with columns: id, stranger_face, exit_time.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, stranger_face, exit_time FROM stranger_exit_logs WHERE DATE(exit_time) = ? ORDER BY exit_time DESC",
                (date.strftime("%Y-%m-%d"),)
            )
            return cursor.fetchall()


    def log_entry(self, employee_id, employee_name, entry_time=None):
        """Record employee entry"""
        with self.get_connection() as conn:
            if entry_time is None:
                entry_time = datetime.datetime.now()
            conn.execute(
                "INSERT INTO entry_logs (employee_id, employee_name, entry_time) VALUES (?, ?, ?)",
                (employee_id, employee_name, entry_time)
            )
            conn.commit()

    def log_exit(self, employee_id, employee_name, exit_time=None):
        """Record employee exit"""
        with self.get_connection() as conn:
            if exit_time is None:
                exit_time = datetime.datetime.now()
            conn.execute(
                "INSERT INTO exit_logs (employee_id, employee_name, exit_time) VALUES (?, ?, ?)",
                (employee_id, employee_name, exit_time)
            )
            conn.commit()

    def delete_employee(self, employee_institute_id):
        """Delete an employee and their logs"""
        with self.get_connection() as conn:
            try:
                cursor = conn.execute("SELECT id FROM employees WHERE employee_institute_id = ?", (employee_institute_id,))
                result = cursor.fetchone()
                if not result:
                    print(f"Employee with ID {employee_institute_id} not found")
                    return False
                
                employee_id = result['id']
                
                conn.execute("DELETE FROM entry_logs WHERE employee_id = ?", (employee_id,))
                conn.execute("DELETE FROM exit_logs WHERE employee_id = ?", (employee_id,))
                conn.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
                
                conn.commit()
                print(f"Employee with ID {employee_institute_id} deleted successfully")
                return True
            except sqlite3.Error as e:
                print(f"Deletion failed: {str(e)}")
                conn.rollback()
                return False

    # -------------------------
    # Stranger-related methods
    # -------------------------


    def get_stranger_photo(self, stranger_number):
        """Retrieve stranger's photo"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT stranger_photo FROM strangers WHERE stranger_number = ?", (stranger_number,))
            result = cursor.fetchone()
            if result:
                return Image.open(io.BytesIO(result['stranger_photo']))
            return None

    def get_stranger_id(self, stranger_number):
        """Get stranger ID from stranger number"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT id FROM strangers WHERE stranger_number = ?", (stranger_number,))
            return cursor.fetchone()

    def get_stranger_details(self, stranger_number):
        with self.get_connection() as conn:
            stranger = conn.execute("SELECT * FROM strangers WHERE stranger_number = ?",
                                    (stranger_number,)).fetchone()
            if not stranger:
                return None

            # Get counts and last logs from stranger_logs table
            entry_count = conn.execute(
                "SELECT COUNT(*) FROM stranger_logs WHERE stranger_id = ? AND log_type = 'entry'",
                (stranger['id'],)
            ).fetchone()
            exit_count = conn.execute(
                "SELECT COUNT(*) FROM stranger_logs WHERE stranger_id = ? AND log_type = 'exit'",
                (stranger['id'],)
            ).fetchone()

            last_entry = conn.execute(
                "SELECT MAX(log_time) FROM stranger_logs WHERE stranger_id = ? AND log_type = 'entry'",
                (stranger['id'],)
            ).fetchone()
            last_exit = conn.execute(
                "SELECT MAX(log_time) FROM stranger_logs WHERE stranger_id = ? AND log_type = 'exit'",
                (stranger['id'],)
            ).fetchone()

            if last_entry and (not last_exit or last_entry > last_exit):
                last_log_type = 'entry'
                last_log_time = last_entry
                current_status = 'entry'
            elif last_exit:
                last_log_type = 'exit'
                last_log_time = last_exit
                current_status = 'exit'
            else:
                last_log_type = None
                last_log_time = None
                current_status = None

            return {
                'stranger_number': stranger['stranger_number'],
                'photo': Image.open(io.BytesIO(stranger['stranger_photo'])),
                'entry_count': entry_count,
                'exit_count': exit_count,
                'last_log_type': last_log_type,
                'last_log_time': last_log_time,
                'current_status': current_status
            }

    def get_stranger_logs_by_date(self, date, log_type=None):
        """Get stranger logs by date. If log_type is provided ('entry' or 'exit'), filter accordingly."""
        with self.get_connection() as conn:
            if log_type:
                cursor = conn.execute(
                    "SELECT stranger_logs.id, strangers.stranger_number, stranger_logs.log_time FROM stranger_logs "
                    "JOIN strangers ON stranger_logs.stranger_id = strangers.id "
                    "WHERE DATE(log_time) = ? AND stranger_logs.log_type = ? ORDER BY log_time DESC",
                    (date.strftime("%Y-%m-%d"), log_type)
                )
            else:
                cursor = conn.execute(
                    "SELECT stranger_logs.id, strangers.stranger_number, stranger_logs.log_time, stranger_logs.log_type FROM stranger_logs "
                    "JOIN strangers ON stranger_logs.stranger_id = strangers.id "
                    "WHERE DATE(log_time) = ? ORDER BY log_time DESC",
                    (date.strftime("%Y-%m-%d"),)
                )
            return cursor.fetchall()

    def delete_stranger_log(self, log_id):
        with self.get_connection() as conn:
            try:
                conn.execute("DELETE FROM stranger_logs WHERE id = ?", (log_id,))
                conn.commit()
                return True
            except sqlite3.Error:
                return False

    def delete_stranger(self, stranger_number):
        """Delete a stranger record and its logs"""
        with self.get_connection() as conn:
            try:
                cursor = conn.execute("SELECT id FROM strangers WHERE stranger_number = ?", (stranger_number,))
                result = cursor.fetchone()
                if not result:
                    print(f"Stranger with number {stranger_number} not found")
                    return False
                stranger_id = result['id']
                conn.execute("DELETE FROM stranger_logs WHERE stranger_id = ?", (stranger_id,))
                conn.execute("DELETE FROM strangers WHERE id = ?", (stranger_id,))
                conn.commit()
                print(f"Stranger with number {stranger_number} deleted successfully")
                return True
            except sqlite3.Error as e:
                print(f"Deletion failed: {str(e)}")
                conn.rollback()
                return False

    


    def save_stranger(self, stranger_number, stranger_photo, stranger_face_embedding):
        """
        Save stranger data to the database.
        Uses INSERT OR IGNORE so that duplicate stranger_number values are skipped.
        """
        with self.get_connection() as conn:
            try:
                # Convert the PIL Image to bytes.
                img_byte_arr = io.BytesIO()
                stranger_photo.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                # Convert the face embedding to bytes.
                embedding_bytes = stranger_face_embedding.tobytes() if hasattr(stranger_face_embedding, 'tobytes') else None
                conn.execute(
                    "INSERT OR IGNORE INTO strangers (stranger_number, stranger_photo, stranger_face_embedding) VALUES (?, ?, ?)",
                    (stranger_number, img_bytes, embedding_bytes)
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                print(f"Error saving stranger {stranger_number}: {e}")
                raise

    def get_stranger_data(self):
        """
        Retrieve all stranger records from the database.
        Returns a list of dictionaries with keys: id, stranger_number, photo, and face_embedding.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, stranger_number, stranger_photo, stranger_face_embedding FROM strangers"
            )
            strangers = []
            for row in cursor:
                stranger = {
                    'id': row['id'],
                    'stranger_number': row['stranger_number'],
                    'photo': Image.open(io.BytesIO(row['stranger_photo'])),
                    'face_embedding': np.frombuffer(row['stranger_face_embedding'], dtype=np.float32) 
                                    if row['stranger_face_embedding'] else None
                }
                strangers.append(stranger)
            return strangers

    def log_stranger_entry(self, stranger_id, log_time=None):
        """
        Log an entry event for a stranger.
        """
        with self.get_connection() as conn:
            if log_time is None:
                log_time = datetime.datetime.now()
            conn.execute(
                "INSERT INTO stranger_logs (stranger_id, log_type, log_time) VALUES (?, ?, ?)",
                (stranger_id, 'entry', log_time)
            )
            conn.commit()

    def log_stranger_exit(self, stranger_id, log_time=None):
        """
        Log an exit event for a stranger.
        """
        with self.get_connection() as conn:
            if log_time is None:
                log_time = datetime.datetime.now()
            conn.execute(
                "INSERT INTO stranger_logs (stranger_id, log_type, log_time) VALUES (?, ?, ?)",
                (stranger_id, 'exit', log_time)
            )
            conn.commit()

    def get_last_stranger_entry(self, stranger_id):
        """
        Retrieve the most recent entry log time for a stranger.
        Returns the log time as a string or None if no entry exists.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(log_time) as last_entry FROM stranger_logs WHERE stranger_id = ? AND log_type = 'entry'",
                (stranger_id,)
            )
            row = cursor.fetchone()
            return row['last_entry'] if row and row['last_entry'] is not None else None

    def get_last_stranger_exit(self, stranger_id):
        """
        Retrieve the most recent exit log time for a stranger.
        Returns the log time as a string or None if no exit exists.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(log_time) as last_exit FROM stranger_logs WHERE stranger_id = ? AND log_type = 'exit'",
                (stranger_id,)
            )
            row = cursor.fetchone()
            return row['last_exit'] if row and row['last_exit'] is not None else None    
    def log_stranger_entry(self, stranger_face, entry_time=None):
        """
        Logs a stranger entry event.
        :param stranger_face: A PIL Image instance representing the unknown face.
        :param entry_time: Optional datetime; defaults to datetime.now()
        """
        with self.get_connection() as conn:
            if entry_time is None:
                entry_time = datetime.datetime.now()
            # Convert face (PIL Image) to bytes
            img_byte_arr = io.BytesIO()
            stranger_face.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            conn.execute(
                "INSERT INTO stranger_entry_logs (stranger_face, entry_time) VALUES (?, ?)",
                (img_bytes, entry_time)
            )
            conn.commit()
    def log_stranger_exit(self, stranger_face, exit_time=None):
        """
        Logs a stranger exit event.
        :param stranger_face: A PIL Image instance representing the unknown face at the time of exit.
        :param exit_time: Optional datetime; defaults to datetime.now()
        """
        with self.get_connection() as conn:
            if exit_time is None:
                exit_time = datetime.datetime.now()
            img_byte_arr = io.BytesIO()
            stranger_face.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            conn.execute(
                "INSERT INTO stranger_exit_logs (stranger_face, exit_time) VALUES (?, ?)",
                (img_bytes, exit_time)
            )
            conn.commit()

    # -------------------------
    # Visitor-related methods
    # -------------------------
    def save_visitor(self, visitor_name, face_embedding, purpose_of_visit, allowed_time, visitor_status="active"):
        """
        Save a visitor's record into the visitors table.
        """
        with self.get_connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO visitors (visitor_name, face_embedding, purpose_of_visit, allowed_time, visitor_status) VALUES (?, ?, ?, ?, ?)",
                    (visitor_name,
                    face_embedding.tobytes() if hasattr(face_embedding, "tobytes") else None,
                    purpose_of_visit,
                    allowed_time,
                    visitor_status)
                )
                conn.commit()
                cursor = conn.execute("SELECT last_insert_rowid()")
                row = cursor.fetchone()
                return row[0]  # Return the newly assigned visitor_id.
            except Exception as e:
                print("Error saving visitor:", e)
                raise

    def log_visitor_entry(self, visitor_id, entry_time=None):
        """
        Log a visitor's entry into the visitor_entry table.
        """
        with self.get_connection() as conn:
            if entry_time is None:
                entry_time = datetime.datetime.now()
            conn.execute(
                "INSERT INTO visitor_entry (visitor_id, entry_time) VALUES (?, ?)",
                (visitor_id, entry_time)
            )
            conn.commit()

    def log_visitor_exit(self, visitor_id, exit_time=None):
        """
        Log a visitor's exit into the visitor_exit table.
        """
        with self.get_connection() as conn:
            if exit_time is None:
                exit_time = datetime.datetime.now()
            conn.execute(
                "INSERT INTO visitor_exit (visitor_id, exit_time) VALUES (?, ?)",
                (visitor_id, exit_time)
            )
            conn.commit()

    def update_visitor_statuses(self):
        """
        Check all visitor records and update the visitor_status.
        If the current time has passed their allowed_time, set status to 'expired';
        otherwise, set to 'active'.
        (Call this function every hour—e.g. via a scheduled job or background thread.)
        """
        with self.get_connection() as conn:
            now = datetime.datetime.now()
            conn.execute(
                "UPDATE visitors SET visitor_status = 'expired' WHERE allowed_time <= ?",
                (now,)
            )
            conn.execute(
                "UPDATE visitors SET visitor_status = 'active' WHERE allowed_time > ?",
                (now,)
            )
            conn.commit()
    def get_active_visitors(self):
        """Retrieve all visitor data"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, visitor_name, face_embedding, purpose_of_visit, allowed_time, visitor_status FROM visitors"
            )
            visitors = []
            for row in cursor:
                visitors.append({
                    'id': row['id'],
                    'visitor_name': row['visitor_name'],
                    'face_embedding': np.frombuffer(row['face_embedding'], dtype=np.float32)
                                    if row['face_embedding'] is not None else None,
                    'purpose_of_visit': row['purpose_of_visit'],
                    'allowed_time': row['allowed_time'],
                    'visitor_status': row['visitor_status']
                })
            return visitors
    def get_visitor_data(self):
        """Retrieve all visitor data"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, visitor_name, face_embedding, purpose_of_visit, allowed_time, visitor_status FROM visitors"
            )
            visitors = []
            for row in cursor:
                visitors.append({
                    'id': row['id'],
                    'visitor_name': row['visitor_name'],
                    'face_embedding': (np.frombuffer(row['face_embedding'], dtype=np.float32)
                                    if row['face_embedding'] is not None else None),
                    'purpose_of_visit': row['purpose_of_visit'],
                    'allowed_time': row['allowed_time'],
                    'visitor_status': row['visitor_status']
                })
            return visitors
        
    def get_visitor_entry_logs_by_date(self, date):
        """
        Retrieve visitor entry logs for a given date.
        Returns rows containing at least: id, visitor_id, entry_time.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, visitor_id entry_time FROM visitor_entry WHERE DATE(entry_time) = ? ORDER BY entry_time DESC",
                (date.strftime("%Y-%m-%d"),)
            )
            return cursor.fetchall()

    def get_visitor_exit_logs_by_date(self, date):
        """
        Retrieve visitor exit logs for a given date.
        Returns rows containing at least: id, visitor_id, exit_time.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, visitor_id, exit_time FROM visitor_exit WHERE DATE(exit_time) = ? ORDER BY exit_time DESC",
                (date.strftime("%Y-%m-%d"),)
            )
            return cursor.fetchall()

if __name__ == "__main__":
    db = DatabaseManager()
    db.initialize_database()