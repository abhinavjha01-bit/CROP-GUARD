import hashlib
from io import BytesIO

import streamlit as st
from PIL import Image
from transformers import pipeline


MODEL_ID = "neghani/plantvillage-mobilenetv2"
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(page_title="Crop Guard", page_icon="🌱")

st.title("🌱 Crop Guard")
st.write("Upload a clear photo of one tomato leaf.")
st.caption(
    "Prototype guidance only. Results are possibilities, "
    "not confirmed diagnoses."
)


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline(
        "image-classification",
        model=MODEL_ID,
        device=-1,
    )


def readable_label(label):
    return label.replace("___", " — ").replace("_", " ")


confirmed_tomato = st.checkbox("I confirm this is a tomato leaf")

photo = st.file_uploader(
    "Upload a tomato leaf photo",
    type=["jpg", "jpeg", "png"],
)

if photo:
    photo_bytes = photo.getvalue()
    photo_id = hashlib.sha256(photo_bytes).hexdigest()

    # Clear the previous result when a different photo is uploaded.
    if st.session_state.get("photo_id") != photo_id:
        st.session_state["photo_id"] = photo_id
        st.session_state.pop("predictions", None)

    try:
        image = Image.open(BytesIO(photo_bytes)).convert("RGB")
        st.image(image, caption="Uploaded leaf", use_container_width=True)

        if st.button("Check leaf", disabled=not confirmed_tomato):
            try:
                with st.spinner("Looking at the leaf..."):
                    st.session_state["predictions"] = load_classifier()(
                        image,
                        top_k=38,
                    )
            except Exception as error:
                st.error(f"Could not load the classifier: {error}")

        if not confirmed_tomato:
            st.info("Confirm that the photo shows a tomato leaf to continue.")

        predictions = st.session_state.get("predictions")

        if predictions:
            top_overall = predictions[0]

            tomato_predictions = [
                item
                for item in predictions
                if item["label"].startswith("Tomato___")
            ]

            if not top_overall["label"].startswith("Tomato___"):
                st.warning(
                    "The model's strongest overall match is not tomato. "
                    "It may have mistaken the crop, so treat the result as uncertain."
                )
                st.caption(
                    f"Strongest overall match: "
                    f"{readable_label(top_overall['label'])} "
                    f"({float(top_overall['score']):.0%})"
                )

            if not tomato_predictions:
                st.warning(
                    "The model found no tomato match. Try a clearer photo "
                    "of one tomato leaf."
                )
            else:
                best_tomato = tomato_predictions[0]
                condition = best_tomato["label"].split("___", 1)[1]
                condition = condition.replace("_", " ")
                score = float(best_tomato["score"])

                st.subheader("Best tomato possibility")
                st.write(f"**{condition}**")
                st.caption(f"Model score: {score:.0%}")

                if (
                    score < CONFIDENCE_THRESHOLD
                    or not top_overall["label"].startswith("Tomato___")
                ):
                    st.warning(
                        "The result is uncertain. Retake the photo or ask a local "
                        "agricultural expert before choosing a treatment."
                    )
                else:
                    st.info("This is a possible match, not a confirmed diagnosis.")

                    if "early blight" in condition.lower():
                        st.markdown("### What early blight can look like")
                        st.write(
                            "It can start as small dark spots on older, lower leaves. "
                            "Larger spots may have ring-like patterns, with yellowing "
                            "around them."
                        )

                        st.markdown("### Lower-cost care steps")
                        st.write(
                            "1. Check older, lower leaves for dark spots with "
                            "ring-like patterns."
                        )
                        st.write(
                            "2. Water at the base of the plant and try to keep "
                            "leaves dry."
                        )
                        st.write(
                            "3. Carefully remove spotted leaves. Clean your tools "
                            "afterward."
                        )
                        st.write(
                            "4. Clear infected plant debris and reduce soil "
                            "splashing onto leaves."
                        )
                        st.caption(
                            "If many leaves are affected, ask an agricultural "
                            "expert before pruning heavily."
                        )

                        st.markdown("### If the problem continues")
                        st.write(
                            "Show the photo and plant to a local KVK or agricultural "
                            "expert. Ask them to confirm the disease and recommend "
                            "treatment suitable for your area and crop stage."
                        )
                        st.caption(
                            "This app does not provide pesticide doses. Use a "
                            "pesticide only if a local expert recommends it, and "
                            "follow the product label and harvest waiting period."
                        )

                        st.markdown("### Sources")
                        st.markdown(
                            "- [University of Minnesota Extension: Early blight "
                            "symptoms and management]"
                            "(https://extension.umn.edu/agriculture/specialty-crops/"
                            "vegetable-farming/disease-management/"
                            "early-blight-in-tomato-and-potato)"
                        )
                        st.markdown(
                            "- [ICAR: Kharif Agro-Advisory for Farmers 2025]"
                            "(https://icar.gov.in/sites/default/files/Circulars/"
                            "ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf)"
                        )
                    else:
                        st.info(
                            "Simple, source-checked guidance for this possible "
                            "condition has not been added yet. Ask a local "
                            "agricultural expert."
                        )

            st.subheader("Why this result?")
            st.write(
                "The model compared the photo with its trained crop and "
                "disease labels."
            )

            with st.expander("See the model's top three overall matches"):
                for item in predictions[:3]:
                    st.write(
                        f"{readable_label(item['label'])}: "
                        f"{float(item['score']):.0%}"
                    )

            st.caption(
                "These scores are model outputs, not verified probabilities. "
                "This version does not highlight disease spots in the image."
            )

            st.subheader("Farmer observations")
            st.write(
                "These answers add context. They do not change the model result yet."
            )

            symptoms = st.multiselect(
                "What do you notice on the leaf?",
                [
                    "Dark or brown spots",
                    "Yellowing",
                    "Curling",
                    "Dry or damaged edges",
                    "Powdery-looking patches",
                    "Not sure",
                ],
            )

            watering = st.selectbox(
                "How has watering been recently?",
                [
                    "About usual",
                    "Less than usual",
                    "More than usual",
                    "Soil is often waterlogged",
                    "Not sure",
                ],
            )

            duration = st.selectbox(
                "When did you first notice the change?",
                [
                    "Today",
                    "In the last few days",
                    "More than a week ago",
                    "Not sure",
                ],
            )

            st.write("**Your notes:**")
            st.write(
                f"- Leaf signs: "
                f"{', '.join(symptoms) if symptoms else 'Not answered'}"
            )
            st.write(f"- Watering: {watering}")
            st.write(f"- Started: {duration}")

            st.caption(
                "This prototype does not save these answers after you close "
                "the app. The observations do not affect the model's result."
            )

    except Exception as error:
        st.error(f"Could not open the uploaded photo: {error}")