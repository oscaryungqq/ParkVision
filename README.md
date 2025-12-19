## ParkVision: A Full-Stack System with Custom-Trained Models for Indoor Parking Detection

![ParkVision Demo GIF](Demo.gif)

This project showcases the end-to-end development of a specialized computer vision solution for the challenging environment of indoor parking lots. It encompasses the entire machine learning lifecycle. **Custom data collection, manual annotation, model fine-tuning**, and deployment throught a **asynchronous backend API**

## The Problem: Why Indoor Parking is a Unique Challenge

Generic, pre-trained object detection models like YOLO often fail in the unique conditions of indoor parking lots due to:

- **Poor and varied lighting conditions.**
- **Ambiguous floor markings** that can be confused with empty parking stalls.
- **Repetitive patterns and lack of distinct landmarks.**
- **Unusual camera angles** from infrastructure cameras or vehicle dashcams.

This project was built to solve this problem for two key users:

1.  **Parking Lot Owners:** To provide an automated, accurate system for monitoring occupancy and managing their facility.
2.  **Drivers:** To enable future applications, such as using a car's front-facing camera to quickly identify available spots.

## Key Features

- **Custom-Trained Deep Computer Vision Models:** Fine-tuned YOLOv8 models that significantly outperform generic object detectors in specialized indoor environments.

- **RESTful API:** A clean, documented API for uploading videos and retrieving processing results.

- **Asynchronous Backend Processing.** Utilizes Celery and Redis to handle long-running video analysis in the background, ensuring the API remains fast and responsive.

- **Scalable Architecture:** A stateless API layer with a stateful persistence layer (PostgreSQL) for futture scaling.

- **Persistent Job Tracking:** All requests are stored as permanent jobs in a PostgreSQL database, allowing users to reliably track their video processing status.

## The Object Detection Pipeline: Custom Models with a Visual Memory

The core of ParkVision is its specialized pipeline. It combines custom-trained "eyes" with a short-term memory" to create a robust understanding of the parking environment.

A generic model is not enough; a custom solution was required.

### 1. Data Collection & Annotation

Recognizing the scarcity of public datasets for this specific niche, I initiated a data collection process:

- **Gathered a targeted dataset** of images and video footage from various indoor parking lots, capturing different lighting conditions, angles, and occupancy levels.

- **Manually annotated hundreds of images**, carefully labeling three distinct classes: `empty_spot`, `occupied_spot`, and `car`. This high-quality, specific data is the foundation of the model's success.

### 2. Fine-Tuning with Transfer Learning

Instead of training a model from scratch, which requires massive amounts of data, I leveraged **transfer learning**:

- I took the powerful, pre-trained **YOLOv8** model, which already understands general shapes and objects.

- I **fine-tuned** this model on my custom-annotated dataset. This process re-trains the model's final layers to become an expert at one specific task: distinguishing between empty and occupied spots in challenging indoor settings.

The result is a set of lightweight, fast, and highly accurate models that excel where generic models fail.

### 3. The "Memory": The Role of DeepSORT for tracking

While the YOLO models act as the system's "eyes" on a frame-by-frame basis, they lack memory. They can't tell if a car in one frame is the same car from the previous frame. This is where **DeepSORT** comes in. It acts as the system's **short-term visual memory**, which is critical for two key reasons:

- **Handling Occlusion:** In a real parking lot, a vehicle's view of a parking spot can be temporarily blocked by a passing car or a pillar. Without a tracker, the system would mistakenly think the spot is now empty. DeepSORT uses a Kalman Filter to **predict an object's position** even when it's momentarily hidden. This allows the system to maintain the "occupied" state correctly, giving it a persistent memory of the object.

- **Ensuring Temporal Stability:** Object detectors can sometimes "flicker," failing to detect an object in a single frame. Without tracking, this would cause the occupancy count to be unstable. DeepSORT ensures that each detected object is assigned a **stable, unique ID**. This application logic relies on this stability; for example, it only marks a spot as occupied after a car with a consistent ID has overlapped with it for a set duration.

By integrating DeepSORT, the system moves beyond simple, stateless detection to a more robust, stateful understanding of the parking lot's activity, significantly reducing errors caused by temporary occlusions and detection flicker.

## System Architecture & Tech Stack

The backend is designed to serve the custom AI models in a scalable and resilient way.

### Tech Stack

- **Backend Framework:** **FastAPI**
- **Asynchronous Task Queue:** **Celery**
- **Message Broker & Cache:** **Redis**
- **Database:** **PostgreSQL** with **SQLAlchemy** (ORM)
- **AI / CV:** **PyTorch**, **Custom Fine-Tuned YOLOv8**, **DeepSORT**, **OpenCV**
- **Containerization:** **Docker** (for services)

```mermaid
graph TD
    A[User] -->|1. POST /upload-video| B(FastAPI Server);
    B -->|2. Create Job Record| C(PostgreSQL DB);
    B -->|3. Send Task(job_id)| D(Redis Queue);
    B -->|4. Return job_id| A;
    E(Celery Worker) -->|5. Fetch Task| D;
    E -->|6. Load Custom AI Models| F[Fine-Tuned YOLOv8];
    E -->|7. Process Video| F;
    E -->|8. Update Job Status in DB| C;
    A -->|9. GET /status/{job_id}| B;
    B -->|10. Read Job Status from DB| C;
    B -->|11. Return Status| A;
```

## Getting Started

Follow these instruction to set up and run the complete application environment locally.

### Prerequisites.

- Python 3.10+
- Git
- Docker (and Docker Compose, optionally) running via Docker Desktop or WSL2.

1. **Install the Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch External Services (PostgreSQL & Redis):**
   _Ensure your Docker daemon is running._

   - **PostgreSQL:**
     ```bash
     docker run --name parkvision-db -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres
     ```
   - **Redis:**
     ```bash
     docker run --name parkvision-redis -p 6379:6379 -d redis
     ```

3. **Initialize the Database Schema (First Time Only):**
   - This command executes the `schema.sql` file inside your running PostgreSQL container, creating the necessary tables and types.
   ```bash
   docker exec -i parkvision-db psql -U postgres < schema.sql
   ```

### Running the Application

You will need **two separate terminals** running simultaneously (in addition to Docker).

1.  **Terminal 1: Start the Celery Worker:**
    _Activate your virtual environment first._

    ```bash
    celery -A worker.celery_app worker --loglevel=info -P solo
    ```

2.  **Terminal 2: Start the FastAPI Server:**
    _Activate your virtual environment first._
    ```bash
    uvicorn api.main:app --reload
    ```

The API is now running and available at `http://127.0.0.1:8000`.

## API Usage & Testing

The easiest way to interact with and test the API is through the auto-generated documentation provided by FastAPI.

**Open your browser and navigate to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

Follow these steps to verify that the entire system is working correctly:

1.  **Upload a Video:**

    - Expand the `POST /api/v1/upload-video/` endpoint.
    - Click "Try it out", choose a small video file, and click "Execute".
    - From the successful response, copy the `job_id`.

2.  **Check Job Status:**

    - Now, use the **real `job_id`** from Step 1.
    - Execute the status check. You should see the status as `PENDING` or `PROCESSING`.
    - Wait until your Celery Worker terminal shows that the job is complete.
    - Execute the status check again. The status should now be `COMPLETED`. Copy the `result_filename` from the response.

3.  **Download the Result:**
    - Expand the `GET /api/v1/download/{filename}` endpoint.
    - Paste the `result_filename` you copied.
    - Execute. Your browser will prompt you to download the processed video.

This confirms the entire end-to-end pipeline is functional.
