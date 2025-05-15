const socket = io();
const video = document.getElementById('webcam');
const processedFrame = document.getElementById('processed-frame');
const recognizedText = document.getElementById('recognized-text');
const saveButton = document.getElementById('save');
const clearButton = document.getElementById('clear');
const toggleCamButton = document.getElementById('toggleCam');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

let isCamOn = false;
let stream = null;
let intervalId = null;
let isSaving = false; // Trạng thái lưu

function startWebcam() {
    if (!isCamOn) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(newStream => {
                stream = newStream;
                video.srcObject = stream;
                video.onloadeddata = () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    isCamOn = true;
                    toggleCamButton.textContent = 'Tắt Cam';
                    intervalId = setInterval(() => {
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                        socket.emit('video_frame', dataUrl);
                    }, 100);
                };
            })
            .catch(err => {
                console.error('Error accessing webcam:', err);
                alert('Không thể truy cập webcam!');
            });
    }
}

function stopWebcam() {
    if (isCamOn && stream) {
        clearInterval(intervalId);
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        isCamOn = false;
        toggleCamButton.textContent = 'Mở Cam';
        processedFrame.src = '';
        processedFrame.style.display = 'none';
    }
}

startWebcam();

toggleCamButton.addEventListener('click', () => {
    if (isCamOn) {
        stopWebcam();
    } else {
        startWebcam();
    }
});

socket.on('response', data => {
    processedFrame.src = data.frame;
    processedFrame.style.display = 'block';
    recognizedText.textContent = data.text;
});

socket.on('clear_response', data => {
    recognizedText.textContent = data.text;
});

socket.on('message', msg => {
    console.log(msg.data);
});

clearButton.addEventListener('click', () => {
    recognizedText.textContent = '';
    socket.emit('clear_text');
});

saveButton.addEventListener('click', () => {
    socket.emit('save_text');

    saveButton.disabled = true;
    saveButton.textContent = 'Đang lưu...';
});

socket.on('save_response', (data) => {
    saveButton.disabled = false;
    saveButton.textContent = 'Lưu Vào File';

    if (data.status === 'success') {
        alert('Đã lưu vào file thành công');

        // ✅ Tạo file .txt và tự động tải về
        const blob = new Blob([data.text], { type: 'text/plain;charset=utf-8' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'recognized_text.txt';
        link.click();
    } else {
        alert('Lỗi khi lưu: ' + data.message);
    }
});

