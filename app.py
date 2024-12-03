from flask import Flask, render_template, request, jsonify
from matplotlib import pyplot as plt
import torch
import torchaudio
from PIL import Image
from torchvision.transforms import transforms
import io
import os
import smtplib
from email.message import EmailMessage

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Updated upload folder
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg','m4a','flac'}
MAIL_PASS = os.getenv("MAIL_PASS")
SENDER_ADDR = os.getenv("SENDER_ADDR")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the pre-trained model
model_path = 'model/Resnet34_2024-05-08--06-37-48_v3.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
# Define a function to transform audio data into images
def transform_data_to_image(audio, sample_rate):
    spectrogram_tensor = (torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64, n_fft=1024)(audio)[0] + 1e-10).log2()
    image_path = os.path.join('spectro_img', 'voice_image.png')
    plt.imsave(image_path, spectrogram_tensor.numpy(), cmap='viridis')
    return image_path

def process_file(filename, model):
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set the model to evaluation mode
    model.eval()

    # Convert to device
    model.to(device)

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((64, 862)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :])
    ])

    # Load the audio
    audio, sample_rate = torchaudio.load(filename)

    # Transform audio to an image and save it
    image_path = transform_data_to_image(audio, sample_rate)

    # Load the saved image and apply transformations
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make predictions using the model
    model.eval()
    with torch.no_grad():
        outputs = model(image.to(device))
    predict = outputs.argmax(dim=1).cpu().detach().numpy().ravel()[0]

    return predict

def send_email(subject, body, to_email):
    sender_email = SENDER_ADDR
    sender_password = MAIL_PASS
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['GET'])
def list_uploads():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        files = [f for f in files if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'audio-file' not in request.files:
        return jsonify({'error': 'No audio file part'})

    audio_file = request.files['audio-file']

    # If user does not select file, browser also
    # submit an empty part without filename
    if audio_file.filename == '':
        return jsonify({'error': 'No selected audio file'})

    # Check if the file extension is allowed
    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Unsupported file extension'})

    # Save the uploaded audio file to the upload folder
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_path = audio_path.replace('\\', '/')

    audio_file.save(audio_path)

    try:
       # Use the uploaded audio file for prediction
        evaluation_result = process_file(audio_path, model)
        evaluation_result_bool = bool(evaluation_result)


        if evaluation_result == 1:
            subject = "Alert: Positive Evaluation Result"
            body = "The evaluation result for the uploaded audio file is positive (1)."
            recipient_email = "unmeshsagar7@gmail.com"
            send_email(subject, body, recipient_email)

        return jsonify({
            'result': str(evaluation_result),
            'evaluation_result': evaluation_result_bool
        })
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
