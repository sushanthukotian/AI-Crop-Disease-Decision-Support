
import os
import io
import csv
import time
import requests

from flask import (
    Blueprint,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    session,
    current_app,
    send_file,
    Response,
)
from werkzeug.utils import secure_filename

from app.models.detection import Detection

detection_bp = Blueprint("detection", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@detection_bp.route("/detect", methods=["GET", "POST"])
def detect():
    current_app.logger.info("DETECT ROUTE ENTERED")

    if "user_id" not in session:
        flash("Please login to detect crop diseases.", "error")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        current_app.logger.info("DETECT POST RECEIVED")

        # Check whether an image was submitted
        if "image" not in request.files:
            flash("No image part. Please select an image.", "error")
            return redirect(request.url)

        file = request.files["image"]

        if not file or file.filename == "":
            flash("Please select an image.", "error")
            return redirect(request.url)

        # Validate extension
        if not allowed_file(file.filename):
            flash(
                "Please upload a JPG, JPEG, or PNG image.",
                "error",
            )
            return redirect(url_for("detection.detect"))

        # Measure uploaded image size
        file.seek(0, 2)
        image_size = file.tell()
        file.seek(0)

        current_app.logger.info(
            "Uploaded image: %s | Size: %.2f MB",
            file.filename,
            image_size / (1024 * 1024),
        )

        # Reject images larger than 10 MB
        if image_size > MAX_IMAGE_SIZE:
            flash(
                "Image is too large. Please upload an image "
                "smaller than 10 MB.",
                "error",
            )
            return redirect(url_for("detection.detect"))

        # Generate a safe unique filename
        filename = secure_filename(file.filename)

        if not filename:
            flash("Invalid image filename.", "error")
            return redirect(url_for("detection.detect"))

        unique_filename = f"{int(time.time())}_{filename}"

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, unique_filename)

        # Save uploaded image
        try:
            file.save(filepath)

            current_app.logger.info(
                "Uploaded image saved successfully: %s",
                filepath,
            )

        except Exception:
            current_app.logger.exception(
                "Failed to save uploaded image"
            )
            flash(
                "Could not save the uploaded image.",
                "error",
            )
            return redirect(url_for("detection.detect"))

        # Run prediction using cloud inference API
        try:
            current_app.logger.info(
                "Sending image to cloud inference API"
            )

            inference_url = os.environ.get("INFERENCE_API_URL")
            inference_key = os.environ.get("INFERENCE_API_KEY")

            if not inference_url or not inference_key:
                raise RuntimeError(
                    "Inference API environment variables are missing"
                )

            with open(filepath, "rb") as image_file:

                response = requests.post(
                    inference_url,
                    headers={
                        "X-API-Key": inference_key
                    },
                    files={
                        "image": (
                            os.path.basename(filepath),
                            image_file,
                            "application/octet-stream",
                        )
                    },
                    timeout=(30, 300),
                )

            current_app.logger.info(
                "Inference API response status: %s",
                response.status_code,
            )

            if response.status_code != 200:
                current_app.logger.error(
                    "Inference API error. Status: %s | Response: %r",
                    response.status_code,
                    response.text[:1000],
                )

                raise RuntimeError(
                    f"Inference API returned HTTP "
                    f"{response.status_code}"
                )

            # Parse prediction response
            prediction = response.json()

            crop = prediction["crop"]
            disease = prediction["disease"]
            confidence = float(prediction["confidence"])
            error_msg = None

            current_app.logger.info(
                "Cloud prediction completed: "
                "%s - %s (%.2f%%)",
                crop,
                disease,
                confidence,
            )

        except requests.exceptions.Timeout:
            current_app.logger.exception(
                "Inference API request timed out"
            )

            flash(
                "Prediction is taking too long. "
                "Please try again after some time.",
                "error",
            )

            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    current_app.logger.exception(
                        "Could not remove uploaded image"
                    )

            return redirect(url_for("detection.detect"))

        except Exception:
            current_app.logger.exception(
                "Cloud crop disease prediction failed"
            )

            flash(
                "Prediction failed. Please try again. "
                "If the problem continues, contact the administrator.",
                "error",
            )

            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    current_app.logger.exception(
                        "Could not remove uploaded image"
                    )

            return redirect(url_for("detection.detect"))

        # Handle prediction validation errors
        if error_msg:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                current_app.logger.exception(
                    "Could not remove invalid uploaded image"
                )

            flash(f"⚠️ {error_msg}", "error")
            return redirect(url_for("detection.detect"))

        # Save prediction to database
        relative_path = f"uploads/{unique_filename}"

        try:
            detection_id = Detection.save_detection(
                session["user_id"],
                crop,
                disease,
                confidence,
                relative_path,
            )

        except Exception:
            current_app.logger.exception(
                "Failed to save detection record"
            )

            flash(
                "Prediction completed, but the result "
                "could not be saved.",
                "error",
            )
            return redirect(url_for("detection.detect"))

        # Show prediction result
        return redirect(
            url_for(
                "detection.result",
                detection_id=detection_id,
            )
        )

    return render_template("detection.html")


@detection_bp.route("/result/<detection_id>")
def result(detection_id):
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    detection_record = Detection.get_detection(detection_id)

    if (
        not detection_record
        or str(detection_record.get("user_id"))
        != str(session["user_id"])
    ):
        flash(
            "Result not found or unauthorized access.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    plots = {}

    try:
        from app.ml.analisis import get_all_plots_base64

        img_full_path = os.path.join(
            current_app.root_path,
            "static",
            detection_record.get("image_path", ""),
        )

        plots = get_all_plots_base64(
            user_id=session["user_id"],
            current_image_path=img_full_path,
            crop=detection_record.get("crop", "Crop"),
            disease=detection_record.get("disease", "Detection"),
            confidence=detection_record.get(
                "confidence",
                0.0,
            ),
        )

    except Exception:
        current_app.logger.exception(
            "Error generating in-memory plots"
        )

    return render_template(
        "result.html",
        detection=detection_record,
        plots=plots,
    )


@detection_bp.route("/download-plot/<detection_id>/<plot_type>")
@detection_bp.route("/download-plot/<detection_id>")
def download_plot(detection_id, plot_type="diagnostic"):
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    detection_record = Detection.get_detection(detection_id)

    if (
        not detection_record
        or str(detection_record.get("user_id"))
        != str(session["user_id"])
    ):
        flash("Unauthorized access.", "error")
        return redirect(url_for("main.dashboard"))

    from app.ml.analisis import get_plot_bytes

    img_full_path = os.path.join(
        current_app.root_path,
        "static",
        detection_record.get("image_path", ""),
    )

    buf = get_plot_bytes(
        plot_type=plot_type,
        user_id=session["user_id"],
        current_image_path=img_full_path,
        crop=detection_record.get("crop", "Crop"),
        disease=detection_record.get("disease", "Detection"),
        confidence=detection_record.get(
            "confidence",
            0.0,
        ),
    )

    if buf is None:
        flash(
            "Unable to generate plot for download.",
            "error",
        )
        return redirect(
            url_for(
                "detection.result",
                detection_id=detection_id,
            )
        )

    plot_names = {
        "diagnostic": "Diagnostic_Feature_Analysis",
        "1": "Monthly_Disease_Ratio",
        "2": "Image_Processing_Results",
        "3": "Crop_Disease_Heatmap",
    }

    name = plot_names.get(
        str(plot_type),
        f"Plot_{plot_type}",
    )

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{name}.png",
    )


@detection_bp.route("/download-csv/<detection_id>")
def download_csv(detection_id):
    if "user_id" not in session:
        return redirect(url_for("auth.login"))

    detection_record = Detection.get_detection(detection_id)

    if (
        not detection_record
        or str(detection_record.get("user_id"))
        != str(session["user_id"])
    ):
        flash("Unauthorized access.", "error")
        return redirect(url_for("main.dashboard"))

    from app.ml.analisis import extract_features

    img_full_path = os.path.join(
        current_app.root_path,
        "static",
        detection_record.get("image_path", ""),
    )

    feats = extract_features(img_full_path) or {}

    si = io.StringIO()
    cw = csv.writer(si)

    headers = [
        "Detection_ID",
        "Timestamp",
        "Crop",
        "Disease",
        "Confidence_Score",
        "Green_Pixel_Ratio",
        "Mean_Hue",
        "Mean_Saturation",
        "Mean_Brightness",
        "Circularity",
        "Aspect_Ratio",
        "Edge_Density",
    ]

    cw.writerow(headers)

    row = [
        str(detection_record.get("_id")),
        str(detection_record.get("timestamp", "")),
        str(detection_record.get("crop", "")),
        str(detection_record.get("disease", "")),
        str(detection_record.get("confidence", "")),
        f"{feats.get('Green Pixel Ratio', 0.0):.2f}",
        f"{feats.get('Mean Hue', 0.0):.2f}",
        f"{feats.get('Mean Saturation', 0.0):.2f}",
        f"{feats.get('Mean Brightness', 0.0):.2f}",
        f"{feats.get('Circularity', 0.0):.2f}",
        f"{feats.get('Aspect Ratio', 0.0):.2f}",
        f"{feats.get('Edge Density', 0.0):.2f}",
    ]

    cw.writerow(row)

    safe_crop = str(
        detection_record.get("crop", "Crop")
    ).replace(" ", "_")

    safe_disease = str(
        detection_record.get("disease", "Disease")
    ).replace(" ", "_")

    output = si.getvalue()
    si.close()

    return Response(
        output,
        mimetype="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment;filename="
                f"Feature_Data_{safe_crop}_{safe_disease}.csv"
            )
        },
    )
