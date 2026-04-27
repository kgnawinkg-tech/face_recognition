import os
import cv2
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

class GenerativeRobustFaceRecognizer:
    def __init__(self, target_size=(100, 100), num_components=50):
        self.target_size = target_size
        self.num_components = num_components
        self.mean_face = None
        self.eigenfaces = None
        self.eigenvalues = None
        self.labels = []
        self.label_map = {}
        
    def _convolution_preprocess(self, image):
        image = cv2.equalizeHist(image)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        processed = cv2.filter2D(image, -1, kernel)
        return processed     
    def load_data(self, data_dir="photos"):
        data = []
        labels = []
        label_id = 0
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory {data_dir}. Please populate it with subfolders of faces.")
            return np.array([]), np.array([])

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        for person_name in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_path):
                continue    
            self.label_map[label_id] = person_name
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    x, y, w, h = faces[0]
                    img = img[y:y+h, x:x+w]
                img = cv2.resize(img, self.target_size)
                img = self._convolution_preprocess(img)
                vector = img.flatten()
                data.append(vector)
                labels.append(label_id)    
            label_id += 1    
        if not data:
            print("No images found in the photos folder. Please add subfolders with images.")
            return np.array([]), np.array([])   
        return np.array(data), np.array(labels)   
    def train(self, data_matrix, labels):
        if len(data_matrix) == 0:
            return    
        self.labels = labels
        n_samples, n_features = data_matrix.shape
        self.mean_face = np.mean(data_matrix, axis=0)
        A = data_matrix - self.mean_face
        print(f"Training on {n_samples} images...")
        U, S, Vt = linalg.svd(A, full_matrices=False)
        self.eigenvalues = (S ** 2) / max(1, (n_samples - 1))
        num_c = min(self.num_components, len(Vt))
        self.eigenfaces = Vt[:num_c]
        self.projected_training_data = np.dot(A, self.eigenfaces.T)
        print("Training complete.")
        
    def generate_synthetic_face(self):
        if self.eigenvalues is None:
            raise ValueError("Model must be trained first.")
        sampled_weights = np.zeros(len(self.eigenvalues[:len(self.eigenfaces)]))
        for i, variance in enumerate(self.eigenvalues[:len(self.eigenfaces)]):
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)
            z0 = np.sqrt(-2.0 * np.log(max(u1, 1e-10))) * np.cos(2.0 * np.pi * u2)
            std_dev = np.sqrt(variance)
            sampled_weight = z0 * std_dev
            sampled_weights[i] = sampled_weight
        synthetic_face_vector = self.mean_face + np.dot(sampled_weights, self.eigenfaces)
        synthetic_face_vector = np.clip(synthetic_face_vector, 0, 255)
        return synthetic_face_vector.astype(np.uint8).reshape(self.target_size)
        
    def recognize_face_newton(self, test_image_vector):
        centered = test_image_vector - self.mean_face
        y = np.dot(centered, self.eigenfaces.T)
        X = self.projected_training_data.T 
        lmbda = 0.1 
        
        num_train = X.shape[1]
        beta = np.zeros(num_train) 
        
        def compute_gradient_hessian(beta):
            residual = y - np.dot(X, beta)
            grad = -2 * np.dot(X.T, residual) + 2 * lmbda * beta
            H = 2 * np.dot(X.T, X) + 2 * lmbda * np.eye(num_train)
            return grad, H
        for _ in range(10): 
            grad, H = compute_gradient_hessian(beta)
            delta = linalg.solve(H, grad)
            beta = beta - delta
            if linalg.norm(delta) < 1e-5:
                break
        class_scores = {}
        for idx, weight in enumerate(beta):
            label = self.labels[idx]
            class_scores[label] = class_scores.get(label, 0) + abs(weight)
        if not class_scores:
            return "Unknown", 0.0
        best_label = max(class_scores, key=class_scores.get)
        total_weight = sum(class_scores.values()) + 1e-9
        confidence_prob = (class_scores[best_label] / total_weight) * 100.0
        reconstructed_face = self.mean_face + np.dot(y, self.eigenfaces)
        corr_matrix = np.corrcoef(test_image_vector, reconstructed_face)
        struct_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix).any() else 0.0
        return self.label_map[best_label], confidence_prob, struct_corr

if __name__ == "__main__":
    recognizer = GenerativeRobustFaceRecognizer()
    data_dir = "photos"
    print("Initializing Math Face Recognition Core...")
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"\nPlease create subfolders inside '{data_dir}' (e.g., '{data_dir}/Alice', '{data_dir}/Bob')")
        print("and populate them with face images, then run this script again.")
    else:
        data, labels = recognizer.load_data(data_dir)
        if len(data) > 0:
            recognizer.train(data, labels)
            mean_face_img = recognizer.mean_face.reshape(recognizer.target_size).astype(np.uint8)
            cv2.imwrite("mean_face.jpg", mean_face_img)
            print("\nSaved calculated 'mean_face.jpg'.")
            print("Generating a mathematical synthetic face using Unit 3 Box-Muller...")
            synth = recognizer.generate_synthetic_face()
            cv2.imwrite("synthetic_face_sample.jpg", synth)
            print("Saved synthetic face to 'synthetic_face_sample.jpg'.")
            test_idx = np.random.randint(0, len(data))
            predicted_name, conf, struct_corr = recognizer.recognize_face_newton(data[test_idx])
            true_name = recognizer.label_map[labels[test_idx]]
            print(f"\n--- Testing Unit 2 Newton Optimization Classifier ---")
            print(f"Test Image True Identity: {true_name}")
            print(f"Predicted Identity (Newton Method): {predicted_name}")
            print(f"Prediction Confidence Score: {conf:.1f}%")
            print(f"Structural Correlation: {struct_corr:.2f} (1.0 is Perfect)")
            print("\n==================================")
            print("AUTOMATED 10-SECOND WEBCAM SCAN")
            print("==================================")
            print("Accessing Mac Webcam... Please look at the camera.")
            print("Scanning for 10 seconds. Do not press any keys. Processing purely in memory (no disk saving)...")            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cap = cv2.VideoCapture(0)            
            import time
            start_time = time.time()
            found_people = {} 
            scan_duration = 10.0         
            while time.time() - start_time < scan_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
                if len(faces) > 0:e
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    x, y, w, h = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, recognizer.target_size)
                    face_processed = recognizer._convolution_preprocess(face_resized)
                    face_vector = face_processed.flatten()
                    name, conf, struct_corr = recognizer.recognize_face_newton(face_vector)
                    if struct_corr < 0.25 or conf < 58.0:
                        name = "Unknown"
                    if name != "Unknown":
                        found_people[name] = found_people.get(name, 0) + 1
                    
            cap.release()
            
            print("\n[Scan Complete] Frame memory cleared. Webcam Shutdown.")
            print("----------------------------------")
            print("PEOPLE DETECTED IN ROOM:")
            valid_people = [name for name, count in found_people.items() if count >= 45]
            
            if not valid_people:
                print(" - No faces could be clearly read.")
            else:
                for person in valid_people:
                    print(f" - {person} (Verified over {found_people[person]} frames)")
            print("----------------------------------\n")
