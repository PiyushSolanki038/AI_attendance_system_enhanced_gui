import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

class AttendanceSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("1000x800")
        self.root.configure(bg='#2C3E50')  # Dark blue background

        # Variables
        self.name_var = tk.StringVar()
        self.enrollment_var = tk.StringVar()
        self.marked_attendance = set()

        # Main Frame
        self.main_frame = ttk.Frame(root, style='Main.TFrame')
        self.main_frame.pack(pady=20, padx=20, fill='both', expand=True)

        # Style Configuration
        style = ttk.Style()
        style.configure('Main.TFrame', background='#34495E')  # Darker blue
        style.configure('Custom.TFrame', background='#ECF0F1')  # Light gray
        style.configure('Custom.TButton', 
                       padding=10, 
                       font=('Arial', 11, 'bold'),
                       background='#3498DB')  # Blue buttons
        style.configure('Title.TLabel', 
                       font=('Arial', 28, 'bold'),
                       foreground='#ECF0F1',
                       background='#34495E')
        style.configure('Header.TLabel', 
                       font=('Arial', 12, 'bold'),
                       foreground='#2C3E50',
                       background='#ECF0F1')

        # Title with enhanced styling
        title_label = ttk.Label(
            self.main_frame, 
            text="Smart Attendance System",
            style='Title.TLabel'
        )
        title_label.pack(pady=20)

        # Registration Frame with shadow effect
        reg_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        reg_frame.pack(pady=10, padx=10, fill='both')

        # Student Details Section
        details_frame = ttk.LabelFrame(
            reg_frame, 
            text="Student Registration", 
            padding=20,
            style='Custom.TFrame'
        )
        details_frame.pack(pady=10, padx=10, fill='both')

        # Name Entry
        ttk.Label(details_frame, text="Student Name:", style='Header.TLabel').pack()
        name_entry = ttk.Entry(
            details_frame, 
            textvariable=self.name_var,
            font=('Arial', 12),
            width=30
        )
        name_entry.pack(pady=5)

        # Enrollment Entry
        ttk.Label(details_frame, text="Enrollment Number:", style='Header.TLabel').pack()
        enrollment_entry = ttk.Entry(
            details_frame, 
            textvariable=self.enrollment_var,
            font=('Arial', 12),
            width=30
        )
        enrollment_entry.pack(pady=5)

        # Buttons Frame
        button_frame = ttk.Frame(reg_frame, style='Custom.TFrame')
        button_frame.pack(pady=20)

        # Enhanced Buttons with Icons
        self.take_img_btn = ttk.Button(
            button_frame,
            text="📸 Take Image",
            command=self.take_image,
            style='Custom.TButton'
        )
        self.take_img_btn.pack(side=tk.LEFT, padx=10)

        self.train_img_btn = ttk.Button(
            button_frame,
            text="🔄 Train Images",
            command=self.train_image,
            style='Custom.TButton'
        )
        self.train_img_btn.pack(side=tk.LEFT, padx=10)

        self.attendance_btn = ttk.Button(
            button_frame,
            text="✓ Take Attendance",
            command=self.take_attendance,
            style='Custom.TButton'
        )
        self.attendance_btn.pack(side=tk.LEFT, padx=10)

        self.check_students_btn = ttk.Button(
            button_frame,
            text="👥 Check Students",
            command=self.show_registered_students,
            style='Custom.TButton'
        )
        self.check_students_btn.pack(side=tk.LEFT, padx=10)

        # Status Frame
        self.status_frame = ttk.LabelFrame(
            self.main_frame, 
            text="System Status", 
            padding=10,
            style='Custom.TFrame'
        )
        self.status_frame.pack(pady=10, padx=10, fill='both')

        self.status_label = ttk.Label(
            self.status_frame, 
            text="System Ready...",
            font=('Arial', 10),
            style='Header.TLabel'
        )
        self.status_label.pack()

        # Initialize directories and CSV
        self.create_directories()
        self.initialize_csv()

    def show_registered_students(self):
        student_window = tk.Toplevel(self.root)
        student_window.title("Registered Students")
        student_window.geometry("600x400")
        student_window.configure(bg='#34495E')

        # Create frame for buttons
        button_frame = ttk.Frame(student_window, style='Custom.TFrame')
        button_frame.pack(pady=10)

        # Create text widget
        text_area = scrolledtext.ScrolledText(
            student_window,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=('Arial', 12),
            bg='#ECF0F1',
            fg='#2C3E50'
        )
        text_area.pack(padx=20, pady=10, fill='both', expand=True)

        def delete_attendance():
            try:
                selected_text = text_area.get(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                messagebox.showwarning("Warning", "Please select a student record to delete")
                return

            enrollment = None
            name = None
            for line in selected_text.split('\n'):
                if line.startswith('Enrollment:'):
                    enrollment = line.split(':')[1].strip()
                elif line.startswith('Name:'):
                    name = line.split(':')[1].strip()

            if enrollment and name:
                # Delete from attendance.csv
                with open('attendance.csv', 'r') as f:
                    lines = f.readlines()
                with open('attendance.csv', 'w') as f:
                    f.write(lines[0])  # Write header
                    for line in lines[1:]:
                        if enrollment not in line:
                            f.write(line)

                # Delete student image
                image_path = os.path.join('student_images', f"{enrollment}_{name}.jpg")
                if os.path.exists(image_path):
                    os.remove(image_path)

                messagebox.showinfo("Success", f"Student {name} deleted successfully")
                show_students()

        def show_students():
            text_area.configure(state='normal')
            text_area.delete(1.0, tk.END)

            # Get active students from attendance.csv
            active_students = set()
            if os.path.exists('attendance.csv'):
                with open('attendance.csv', 'r') as f:
                    next(f)  # Skip header
                    for line in f:
                        name, enrollment, *_ = line.strip().split(',')
                        active_students.add((name, enrollment))

            if os.path.exists('student_images'):
                text_area.insert(tk.END, "Active Students:\n\n")
                for student in active_students:
                    name, enrollment = student
                    text_area.insert(tk.END, f"Name: {name}\nEnrollment: {enrollment}\n\n")
            else:
                text_area.insert(tk.END, "No registered students found.")
            
            text_area.configure(state='disabled')

        # Add Delete Button
        delete_btn = ttk.Button(
            button_frame,
            text="🗑️ Delete Student",
            command=delete_attendance,
            style='Custom.TButton'
        )
        delete_btn.pack(side=tk.LEFT, padx=5)

        # Add Refresh Button
        refresh_btn = ttk.Button(
            button_frame,
            text="🔄 Refresh List",
            command=show_students,
            style='Custom.TButton'
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)

        # Show initial student list
        show_students()

    def create_directories(self):
        os.makedirs('student_images', exist_ok=True)
        os.makedirs('training_data', exist_ok=True)

    def initialize_csv(self):
        if not os.path.exists('attendance.csv'):
            with open('attendance.csv', 'w') as f:
                f.write('Name,Enrollment,Date,Time,Status\n')

    def take_image(self):
        if not self.name_var.get() or not self.enrollment_var.get():
            messagebox.showerror("Error", "Please enter both name and enrollment number!")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        image_captured = False
        
        while not image_captured:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Cannot access camera!")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Take Image (Press SPACE to capture)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if len(faces) > 0:
                    student_img_path = os.path.join('student_images', 
                        f"{self.enrollment_var.get()}_{self.name_var.get()}.jpg")
                    cv2.imwrite(student_img_path, frame)
                    image_captured = True
                    self.status_label.config(
                        text=f"Image captured for {self.name_var.get()} ({self.enrollment_var.get()})"
                    )
                else:
                    messagebox.showwarning("Warning", "No face detected! Please try again.")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        if image_captured:
            messagebox.showinfo("Success", "Image captured successfully!")
            self.mark_attendance(self.name_var.get(), self.enrollment_var.get())

    def mark_attendance(self, name, enrollment):
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        
        with open('attendance.csv', 'a') as f:
            f.write(f"{name},{enrollment},{date_string},{time_string},Present\n")

    def train_image(self):
        if not os.listdir('student_images'):
            messagebox.showerror("Error", "No images found for training!")
            return

        self.status_label.config(text="Training images... Please wait...")
        self.root.update()

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = []
        ids = []
        id_mapping = {}
        
        for idx, img_file in enumerate(os.listdir('student_images')):
            if img_file.endswith('.jpg'):
                enrollment = img_file.split('_')[0]
                id_mapping[enrollment] = idx
                
                img_path = os.path.join('student_images', img_file)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in detected_faces:
                    faces.append(gray[y:y+h, x:x+w])
                    ids.append(idx)

        if not faces:
            messagebox.showerror("Error", "No faces detected in images!")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        
        recognizer.write('training_data/trainer.yml')
        np.save('training_data/id_mapping.npy', id_mapping)
        
        self.status_label.config(text="Training completed successfully!")
        messagebox.showinfo("Success", "Training completed!")

    def take_attendance(self):
        if not os.path.exists('training_data/trainer.yml'):
            messagebox.showerror("Error", "Please train the images first!")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('training_data/trainer.yml')
        
        id_mapping = np.load('training_data/id_mapping.npy', allow_pickle=True).item()
        rev_mapping = {v: k for k, v in id_mapping.items()}
        
        student_details = {}
        for img_file in os.listdir('student_images'):
            if img_file.endswith('.jpg'):
                enrollment = img_file.split('_')[0]
                name = '_'.join(img_file.split('_')[1:]).split('.')[0]
                student_details[enrollment] = name

        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                
                try:
                    id_, conf = recognizer.predict(roi_gray)
                    if conf < 70:
                        enrollment = rev_mapping.get(id_)
                        name = student_details.get(enrollment, "Unknown")
                        
                        cv2.putText(frame, f"{name}", (x, y-10), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        if enrollment not in self.marked_attendance:
                            self.mark_attendance(name, enrollment)
                            self.marked_attendance.add(enrollment)
                            self.status_label.config(text=f"Marked attendance for {name}")
                            messagebox.showinfo("Attendance Marked", 
                                             f"Attendance marked successfully for {name}")
                except:
                    pass

            cv2.imshow('Attendance System (Press Q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemGUI(root)
    root.mainloop()

