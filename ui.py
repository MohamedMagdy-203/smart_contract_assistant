import gradio as gr
import requests
import os

# رابط الـ API (تأكد إن السيرفر شغال على نفس البورت)
API_URL = "http://127.0.0.1:8000"

def process_file(file):
    """
    دالة لرفع الملف للسيرفر
    """
    if not file:
        return "⚠️ Please select a file first."
    
    # Gradio sometimes passes the file object wrapper, we need the path
    file_path = file.name if hasattr(file, 'name') else file
    
    try:
        # بنجهز الملف عشان نبعته للـ API
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            return "✅ File uploaded and processed successfully! You can ask now."
        else:
            return f"❌ Error: {response.text}"
            
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

def ask_question(message, history):
    """
    دالة الشات: Gradio بيبعت الرسالة والـ History تلقائياً
    """
    if not message:
        return ""
    
    try:
        # بنكلم الـ API بتاعنا
        payload = {"query": message}
        response = requests.post(f"{API_URL}/ask", json=payload)
        
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        else:
            return f"Error from Server: {response.text}"
            
    except Exception as e:
        return f"Connection Error: Is the backend running? \nDetails: {str(e)}"

# --- بناء الواجهة ---
# التعديل هنا: الـ Theme بيتحط في الـ Blocks بس
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Smart Contract Assistant")
    gr.Markdown("Upload a PDF contract and ask questions about it.")
    
    with gr.Row():
        # زرار الرفع
        file_input = gr.File(label="Upload Contract (PDF)", file_types=[".pdf"])
        upload_status = gr.Textbox(label="Status", interactive=False)
    
    # ربط زرار الرفع بدالة المعالجة
    file_input.upload(fn=process_file, inputs=file_input, outputs=upload_status)
    
    # واجهة الشات (شلنا منها theme="soft" عشان هي بتورثه من اللي فوق)
    gr.ChatInterface(
        fn=ask_question,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask me anything about the contract...", container=False, scale=7),
    )

if __name__ == "__main__":
    demo.launch()