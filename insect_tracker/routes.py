import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, send_from_directory
from insect_tracker import app, db, bcrypt
from insect_tracker.forms import RegistrationForm, LoginForm, UpdateAccountForm, UploadTrapImage
from insect_tracker.models import User
from flask_login import login_user, current_user, logout_user, login_required
from utils.inference_pipeline import run_inference  # Your refactored logic
import uuid


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('upload'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

def save_trap_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    folder = os.path.join(app.root_path, 'static', 'trap_images')
    os.makedirs(folder, exist_ok=True)
    picture_path = os.path.join(folder, picture_fn)

    i = Image.open(form_picture)
    i = i.convert("RGB")  # Avoid alpha channel issues
    i.save(picture_path, quality=100)  # ❗ NO compression

    return picture_fn

@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)

@app.route("/upload", methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadTrapImage()
    if form.validate_on_submit():
        if form.picture.data:
            import shutil  # ✅ You can place this at the top of the file instead

            # === 1. Clean up old trap images ===
            trap_folder = os.path.join(app.root_path, 'static', 'trap_images')
            os.makedirs(trap_folder, exist_ok=True)
            for f in os.listdir(trap_folder):
                os.remove(os.path.join(trap_folder, f))

            # === 2. Clean up previous output folders ===
            output_root = os.path.join(app.root_path, 'static', 'output')
            os.makedirs(output_root, exist_ok=True)
            for subdir in os.listdir(output_root):
                subpath = os.path.join(output_root, subdir)
                if os.path.isdir(subpath):
                    shutil.rmtree(subpath)

            # === 3. Save new uploaded trap image ===
            original_filename = form.picture.data.filename
            picture_file = save_trap_picture(form.picture.data)
            image_path = os.path.join(trap_folder, picture_file)

            # === 4. Create new output session directory ===
            session_id = str(uuid.uuid4())
            output_dir = os.path.join(output_root, session_id)
            os.makedirs(output_dir, exist_ok=True)

            # === 5. Run inference pipeline ===
            results = run_inference(image_path=image_path, output_dir=output_dir, original_filename=original_filename)

            # === 6. Pass results to the template ===
            return render_template('upload.html',
                                   title='Image Upload',
                                   form=form,
                                   processed=True,
                                   class_counts=results['class_counts'],
                                   annotated_img=url_for('static', filename=f'output/{session_id}/annotated_output.jpg'),
                                   summary_csv=url_for('download_file', filename=f'output/{session_id}/class_summary.csv'),
                                   detailed_csv=url_for('download_file', filename=f'output/{session_id}/detailed_predictions.csv'),
                                   zip_path=url_for('download_file', filename=f'output/{session_id}/results.zip'),
                                   coco_json=url_for('download_file', filename=f'output/{session_id}/coco_annotations.json'))

    return render_template('upload.html', title='Image Upload', form=form)


@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename, as_attachment=True)
