from app import app
from inference import get_prediction
from commons import format_class_name
from flask import Flask, flash, request, redirect, url_for, render_template

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_name=class_name)
    return render_template('index.html')