# BookHawk: Digitize Your Bookshelf 📚

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BookHawk is an intelligent application that uses computer vision and large language models to automatically identify and catalog books from an image of your bookshelf. Simply upload a photo, and BookHawk will detect each book, extract its title and author, and present the information in a clean, organized format.

## ✨ Features

- **Automatic Book Detection**: Utilizes the YOLOv10 model to accurately locate books in an image.
- **Precise Segmentation**: Employs the SAM (Segment Anything Model) to isolate each detected book from its background.
- **Intelligent Information Extraction**: Leverages a powerful multimodal large language model (Ollama with Qwen) to perform OCR and extract the book's title and author.
- **Interactive Web Interface**: A user-friendly frontend built with React to upload images and view results in real-time.
- **Streaming API**: A FastAPI backend that streams results as they are processed, providing a responsive user experience.

## 🚀 Demo

1.  **Upload an image of your bookshelf.**
<img width="667" height="204" alt="Screenshot 2025-09-28 at 15 39 25" src="https://github.com/user-attachments/assets/cedef6e8-3ab6-40d1-a988-815096013f4c" />


2.  **BookHawk detects and classifies each book.**
<img width="1279" height="463" alt="Screenshot 2025-09-28 at 15 39 52" src="https://github.com/user-attachments/assets/2ca16109-b1b8-40f6-bab8-1823a6797a71" />


3.  **View detailed information for each book.**
<img width="1219" height="717" alt="Screenshot 2025-09-28 at 15 40 28" src="https://github.com/user-attachments/assets/80f4e490-c465-4a99-80a1-8e7d359b7874" />


## 🛠️ Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: React.js
- **Computer Vision**:
    - YOLOv10 for object detection
    - Segment Anything Model (SAM) for segmentation
- **LLM**: Ollama with a multimodal model (e.g., Qwen) for OCR and data extraction
- **Deployment**: Docker (optional)

## ⚙️ Getting Started

### Prerequisites

- Python 3.9+
- An available Ollama instance with a multimodal model.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/michaelzm/bookhawk.git
    cd bookhawk
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure your environment:**
    Create a `.env` file in the root directory and add the following, adjusting the values for your setup:
    ```
    OLLAMA_HOST="http://<your-ollama-host-ip>:11434"
    MODEL="qwen2.5vl:7b"
    PROMPT="You will receive a image of a book front view in a book shelf. Extract the author and title of the book image. Return the output as JSON. Output the JSON as {'author': .. , 'title': .., 'language': ..}. Do not mistake the genre or publisher as the author. If you cannot extract the author from the image, output the author that has published the book that you know of."
    ```

### Running the Application

1.  **Start the FastAPI server:**
    To run the server and make it accessible on your local network, use:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8000` or `http://<your-local-ip>:8000` from another device on your network.

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
