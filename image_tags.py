import os
import shutil
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk


class ImageOrganizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Tags Tool")

        self.root_folder = filedialog.askdirectory(title="Select Directory")
        if not self.root_folder:
            messagebox.showerror("Tips", "No folder selected")
            root.destroy()
            return

        self.source_folder = os.path.join(self.root_folder, 'source')
        if not os.path.exists(self.source_folder):
            messagebox.showerror("Tips", "Source folder does not exist")
            root.destroy()
            return

        self.image_files = self.get_image_files(self.source_folder)
        if not self.image_files:
            messagebox.showinfo("Tips", "No images found in the source folder")
            root.destroy()
            return

        self.current_image_index = 0
        self.last_moved_image = None  # Track the last moved image and its destination

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.remaining_label = tk.Label(root, text=f"Remaining photos: {len(self.image_files)}")
        self.remaining_label.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.buttons = sorted(self.get_subfolders(self.root_folder))
        self.shortcut_keys = "1234567890abcdefghijklmnopqrstuvwxyz"  # Define shortcut keys
        self.bindings = {}  # Store the mapping of shortcut keys to button names

        for i, button_name in enumerate(self.buttons):
            if i < len(self.shortcut_keys):
                shortcut_key = self.shortcut_keys[i]
                button = tk.Button(self.button_frame, text=f"{button_name} ({shortcut_key})",
                                   command=lambda name=button_name: self.copy_image(name))
                button.pack(side=tk.LEFT)
                self.bindings[shortcut_key] = button_name
        self.root.bind_all("<Key>", self.handle_keypress)

        self.show_image()

    def handle_keypress(self, event):
        key = event.keysym.lower()
        print(f"Key pressed: {key}")  # Debugging line to see key
        if key in self.bindings:
            self.copy_image(self.bindings[key])
        elif key == 'z':
            self.undo_last_action()
        else:
            # Handle numeric keypad keys
            if key.startswith("kp_"):
                num_key = key[-1]  # Get the last character which should be the number
                if num_key in self.bindings:
                    self.copy_image(self.bindings[num_key])

    def get_image_files(self, folder):
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(image_extensions)]

    def get_subfolders(self, folder):
        return [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f != 'source']

    def show_image(self):
        if self.image_files:
            image_path = self.image_files[self.current_image_index]
            image = Image.open(image_path)
            original_width, original_height = image.size
            if original_height != 600:
                new_height = 600
                new_width = int((new_height / original_height) * original_width)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def copy_image(self, folder_name):
        if self.image_files:
            source_image_path = self.image_files[self.current_image_index]
            destination_folder = os.path.join(self.root_folder, folder_name)
            destination_image_path = os.path.join(destination_folder, os.path.basename(source_image_path))
            shutil.move(source_image_path, destination_image_path)
            self.last_moved_image = (destination_image_path, source_image_path)  # Save the last move
            self.image_files.pop(self.current_image_index)  # Remove the processed file
            self.update_remaining_label()  # Update the remaining photos label
            if not self.image_files:
                messagebox.showinfo("Tips", "All files have been processed")
                self.root.quit()  # Exit the program
            else:
                self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
                self.show_image()
        else:
            messagebox.showinfo("Tips", "No images")

    def undo_last_action(self):
        if self.last_moved_image:
            destination_image_path, source_image_path = self.last_moved_image
            shutil.move(destination_image_path, source_image_path)
            self.image_files.insert(self.current_image_index, source_image_path)  # Reinsert the image
            self.update_remaining_label()  # Update the remaining photos label
            self.show_image()  # Refresh the current image
            self.last_moved_image = None  # Clear the last move

    def update_remaining_label(self):
        self.remaining_label.config(text=f"Remaining photos: {len(self.image_files)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageOrganizer(root)
    root.mainloop()
