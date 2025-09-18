import tkinter as tk
from tkinter import messagebox

import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document


def load_data(file_path):
    """Loads all documents from a Chroma DB and returns as a list of dicts with 'text' and 'target'."""
    persist_dir = "../reddit-chroma-db"
    ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")
    try:
        db = Chroma(persist_directory=persist_dir,
                    embedding_function=ollama_emb)
        results = db.get(include=["documents", "metadatas"])
        data = []
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        for doc, meta in zip(docs, metas):
            entry = {"text": doc}
            if meta and isinstance(meta, dict):
                entry.update(meta)
            data.append(entry)
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load from Chroma DB: {e}")
        return None


def update_chroma_target(db, link, new_target):
    """Update the 'target' field in metadata for the document with the given link."""
    # Find the document by its metadata link
    matches = db.get(where={"link": link}, include=["documents", "metadatas"])
    docs = matches.get("documents", [])
    metas = matches.get("metadatas", [])
    if docs:
        db.delete(where={"link": link})
        for doc, meta in zip(docs, metas):
            if not meta:
                meta = {}
            meta["target"] = new_target
            db.add_documents([Document(page_content=doc, metadata=meta)])


class LabelingApp:
    def __init__(self, root, file_path, startingIndex=0):
        self.persist_dir = "../reddit-chroma-db"
        self.ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")
        self.db = Chroma(persist_directory=self.persist_dir,
                         embedding_function=self.ollama_emb)
        self.root = root
        self.root.title("Text Labeler")
        font = ("Arial", 18)
        self.root.option_add("*Font", font)

        self.file_path = file_path
        self.data = load_data(file_path)
        if self.data is None:
            self.root.destroy()
            return

        self.current_index = startingIndex

        # UI Elements
        self.text_label = tk.Label(
            root, text="", font=('Arial', 14))
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
        self.root.quit()

    def display_current_row(self):
        if self.current_index < len(self.data):
            row = self.data[self.current_index]
            # Set the label to the title if available, else blank
            self.text_label.config(text=row.get("title", ""))
            self.text_box.config(state='normal')
            self.text_box.delete('1.0', tk.END)
            self.text_box.insert('1.0', row.get("text", "No text found"))
            self.text_box.config(state='disabled')
            self.status_label.config(
                text=f"Progress: {self.current_index + 1}/{len(self.data)}")
            self.text_box.config(font=('Arial', 18))
        else:
            messagebox.showinfo(
                "Done", "You have finished labeling all the examples.")
            self.root.quit()

    def on_yes_click(self):
        self.data[self.current_index]['target'] = 'positive'
        link = self.data[self.current_index].get('link')
        if link:
            update_chroma_target(self.db, link, 'positive')
        self.current_index += 1
        self.display_current_row()

    def on_no_click(self):
        self.data[self.current_index]['target'] = 'negative'
        link = self.data[self.current_index].get('link')
        if link:
            update_chroma_target(self.db, link, 'negative')
        self.current_index += 1
        self.display_current_row()


# Main application loop
if __name__ == "__main__":
    # file_name is not used, but kept for compatibility
    file_name = "../reddit-chroma-db"  # Not used, but required by LabelingApp
    root = tk.Tk()
    app = LabelingApp(root, file_name, startingIndex=0)
    root.mainloop()
