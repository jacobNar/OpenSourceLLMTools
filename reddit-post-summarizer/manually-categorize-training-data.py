import tkinter as tk
from tkinter import messagebox
import json


def load_data(file_path):
    """Loads a list of dictionaries from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        messagebox.showerror("Error", f"The file '{file_path}' was not found.")
        return None
    return data


def save_data(data, file_path):
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


class LabelingApp:
    def __init__(self, root, file_path):
        self.root = root
        self.root.title("Text Labeler")

        self.file_path = file_path
        self.data = load_data(file_path)
        if self.data is None:
            self.root.destroy()
            return

        self.current_index = 90

        # UI Elements
        self.text_label = tk.Label(
            root, text="Text to classify:", font=('Arial', 14))
        self.text_label.pack(pady=5)

        self.text_box = tk.Text(root, height=10, width=80,
                                wrap='word', font=('Arial', 12))
        self.text_box.pack(padx=10, pady=5)
        self.text_box.config(state='disabled')  # Make it read-only

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.yes_button = tk.Button(
            self.button_frame, text="Positive", bg='green', fg='white', command=self.on_yes_click)
        self.yes_button.pack(side='left', padx=10)

        self.no_button = tk.Button(
            self.button_frame, text="Negative", bg='red', fg='white', command=self.on_no_click)
        self.no_button.pack(side='right', padx=10)

        self.quit_button = tk.Button(
            root, text="Quit", bg='gray', fg='white', command=self.on_quit)
        self.quit_button.pack(pady=10)

        self.status_label = tk.Label(
            root, text="", bd=1, relief=tk.SUNKEN, anchor='w')
        self.status_label.pack(fill=tk.X, side='bottom')

        self.display_current_row()

    def on_quit(self):
        if self.current_index < len(self.data):
            messagebox.showinfo(
                "Warning", "You have unsaved changes. Do you want to save?")
            save_response = messagebox.askyesno(
                "Save?", "Do you want to save your changes?")
            if save_response:
                save_data(self.data, self.file_path)
                self.root.quit()
            else:
                self.root.quit()
        else:
            messagebox.showinfo(
                "Done", "You have finished labeling all the examples. The file has been updated.")
            save_data(self.data, self.file_path)
            self.root.quit()

    def display_current_row(self):
        if self.current_index < len(self.data):
            row = self.data[self.current_index]
            self.text_box.config(state='normal')
            self.text_box.delete('1.0', tk.END)
            self.text_box.insert('1.0', row.get("text", "No text found"))
            self.text_box.config(state='disabled')
            self.status_label.config(
                text=f"Progress: {self.current_index + 1}/{len(self.data)}")
        else:
            messagebox.showinfo(
                "Done", "You have finished labeling all the examples. The file has been updated.")
            save_data(self.data, self.file_path)
            self.root.quit()

    def on_yes_click(self):
        self.data[self.current_index]['target'] = 'positive'
        self.current_index += 1
        self.display_current_row()

    def on_no_click(self):
        self.data[self.current_index]['target'] = 'negative'
        self.current_index += 1
        self.display_current_row()


# Main application loop
if __name__ == "__main__":
    file_name = "training-data.jsonl"  # Name of your file
    root = tk.Tk()
    app = LabelingApp(root, file_name)
    root.mainloop()
