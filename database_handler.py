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
