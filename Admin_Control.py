import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from config import CONFIG
from face_processor import FaceProcessor
from database_handler import DatabaseManager
from employee_registrar import EmployeeRegistrar
import datetime

class AdminApp:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.db = DatabaseManager()
        self.employee_registrar = EmployeeRegistrar()

    def run(self):
        st.title("Employee Management System - Admin Panel")

        menu = ["View Employees", "Register Employee", "Delete Employee", "View Logs", "Manage Logs", "Manual Log Entry"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "View Employees":
            self.view_employees()
        elif choice == "Register Employee":
            self.register_employee()
        elif choice == "Delete Employee":
            self.delete_employee()
        elif choice == "View Logs":
            self.view_logs()
        elif choice == "Manage Logs":
            self.manage_logs()
        elif choice == "Manual Log Entry":
            self.manual_log_entry()

    def view_employees(self):
        st.header("Registered Employees")
        employees = self.db.get_employee_data()
        
        search_id = st.text_input("Search by Employee Institute ID")
        filtered_employees = [emp for emp in employees if search_id.lower() in emp['employee_institute_id'].lower()]
        
        for emp in filtered_employees:
            if st.button(f"{emp['name']} (ID: {emp['employee_institute_id']})"):
                self.show_employee_details(emp['employee_institute_id'])

    def show_employee_details(self, employee_institute_id):
        emp = self.db.get_employee_details(employee_institute_id)
        st.subheader(f"Details for {emp['name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(emp['photo'], width=200)
        with col2:
            st.write(f"Institute ID: {emp['employee_institute_id']}")
            st.write(f"Name: {emp['name']}")
            st.write(f"Entry Count: {emp['entry_count']}")
            st.write(f"Exit Count: {emp['exit_count']}")
            st.write(f"Last Log: {emp['last_log_type']} at {emp['last_log_time']}")
            st.write(f"Current Status: {'Inside Campus' if emp['current_status'] == 'entry' else 'Outside Campus'}")
            
            if emp['last_log_time']:
                last_log_time = datetime.datetime.strptime(emp['last_log_time'], "%Y-%m-%d %H:%M:%S.%f")
                time_since_last_log = datetime.datetime.now() - last_log_time
                
                # Calculate hours and minutes
                total_seconds = int(time_since_last_log.total_seconds())
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Format the output
                time_display = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
                st.write(f"Time since last log: {time_display}")




    def register_employee(self):
        st.header("Register New Employee")
        name = st.text_input("Employee Name")
        institute_id = st.text_input("Employee Institute ID")

        registration_method = st.radio("Registration Method", ["Upload Photos", "Live Capture"])

        if registration_method == "Upload Photos":
            uploaded_files = st.file_uploader("Upload 10 face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            if st.button("Register") and name and institute_id and len(uploaded_files) == 10:
                self.process_uploaded_photos(name, institute_id, uploaded_files)
        else:
            if st.button("Start Live Registration") and name and institute_id:
                self.employee_registrar.capture_face_samples(name, institute_id)
                st.success(f"Live registration completed for {name}")

    def process_uploaded_photos(self, name, institute_id, uploaded_files):
        embeddings = []
        for file in uploaded_files:
            image = Image.open(file)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            embeds = self.face_processor.get_embeddings(cv_image)
            if embeds:
                embeddings.append(embeds[0])

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding /= np.linalg.norm(avg_embedding)
            try:
                self.db.save_employee(institute_id, name, avg_embedding, Image.open(uploaded_files[0]))
                st.success(f"Successfully registered {name}")
            except Exception as e:
                st.error(f"Failed to register employee: {str(e)}")
        else:
            st.error("Failed to generate embeddings. Please try again.")

    def manual_log_entry(self):
        st.header("Manual Log Entry")
        institute_id = st.text_input("Employee Institute ID")
        log_type = st.radio("Log Type", ["Entry", "Exit"])
        date = st.date_input("Log Date", value=datetime.date.today())
        time = st.time_input("Log Time", value=datetime.datetime.now().time())
        log_time = datetime.datetime.combine(date, time)


        if st.button("Add Log Entry"):
            employee_ID = self.db.get_employee_id(institute_id)['id']
            name = self.db.get_employee_name(institute_id)['name']
            if employee_ID:
                if log_type == "Entry":
                    self.db.log_entry(employee_ID, name)
                else:
                    self.db.log_exit(employee_ID, name)
                st.success(f"{log_type} log added successfully for {name}")
            else:
                st.error("Employee not found")

    def delete_employee(self):
        st.header("Delete Employee")
        employees = self.db.get_employee_data()
        
        search_id = st.text_input("Search by Employee Institute ID")
        filtered_employees = [emp for emp in employees if search_id.lower() in emp['employee_institute_id'].lower()]
        
        selected_employees = []
        for emp in filtered_employees:
            if st.checkbox(f"{emp['name']} (ID: {emp['employee_institute_id']})"):
                selected_employees.append(emp['employee_institute_id'])
        
        if st.button("Delete Selected Employees"):
            for emp_id in selected_employees:
                if self.db.delete_employee(emp_id):
                    st.success(f"Employee with ID {emp_id} deleted successfully")
                else:
                    st.error(f"Failed to delete employee with ID {emp_id}")

    def view_logs(self):
        st.header("View Logs")
        log_type = st.radio("Select Log Type", ["Entry Logs", "Exit Logs"])
        date = st.date_input("Select Date")
        
        if log_type == "Entry Logs":
            logs = self.db.get_entry_logs_by_date(date)
            self.display_logs(logs, "Entry")
        else:
            logs = self.db.get_exit_logs_by_date(date)
            self.display_logs(logs, "Exit")

    def display_logs(self, logs, log_type):
        for log in logs:
            st.write(f"{log_type} - Employee: {log[1]}, Time: {log[2]}")

    def manage_logs(self):
        st.header("Manage Logs")
        log_type = st.radio("Select Log Type", ["Entry Logs", "Exit Logs"])
        date = st.date_input("Select Date")
        
        if log_type == "Entry Logs":
            logs = self.db.get_entry_logs_by_date(date)
            selected_logs = self.select_logs(logs, "Entry")
        else:
            logs = self.db.get_exit_logs_by_date(date)
            selected_logs = self.select_logs(logs, "Exit")
        
        if st.button("Delete Selected Logs"):
            for log_id in selected_logs:
                if log_type == "Entry Logs":
                    if self.db.delete_entry_log(log_id):
                        st.success(f"Entry log {log_id} deleted successfully")
                    else:
                        st.error(f"Failed to delete entry log {log_id}")
                else:
                    if self.db.delete_exit_log(log_id):
                        st.success(f"Exit log {log_id} deleted successfully")
                    else:
                        st.error(f"Failed to delete exit log {log_id}")

    def select_logs(self, logs, log_type):
        selected_logs = []
        for log in logs:
            if st.checkbox(f"{log_type} - Employee: {log[1]}, Time: {log[2]}", key=log[0]):
                selected_logs.append(log[0])
        return selected_logs

if __name__ == "__main__":
    app = AdminApp()
    app.run()
