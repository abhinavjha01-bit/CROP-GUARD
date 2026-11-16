import hashlib
from io import BytesIO

import streamlit as st
from PIL import Image
from transformers import pipeline


MODEL_ID = "neghani/plantvillage-mobilenetv2"
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(page_title="Crop Guard", page_icon="🌱")

selected_language = st.selectbox(
    "Choose language / भाषा चुनें / भाषा निवडा",
    ["English", "हिन्दी", "मराठी"],
)


def tr(english, hindi, marathi):
    if selected_language == "हिन्दी":
        return hindi
    if selected_language == "मराठी":
        return marathi
    return english


@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline(
        "image-classification",
        model=MODEL_ID,
        device=-1,
    )


def readable_label(label):
    return label.replace("___", " — ").replace("_", " ")


st.title(tr("🌱 Crop Guard", "🌱 क्रॉप गार्ड", "🌱 क्रॉप गार्ड"))
st.write(
    tr(
        "Upload a clear photo of one tomato leaf.",
        "टमाटर के एक पत्ते की साफ़ फ़ोटो अपलोड करें।",
        "टोमॅटोच्या एका पानाचा स्पष्ट फोटो अपलोड करा.",
    )
)
st.caption(
    tr(
        "Prototype guidance only. Results are possibilities, not confirmed diagnoses.",
        "यह शुरुआती नमूना केवल मार्गदर्शन देता है। परिणाम संभावनाएँ हैं, पक्का निदान नहीं।",
        "हा प्राथमिक नमुना फक्त मार्गदर्शन देतो. हे संभाव्य निष्कर्ष आहेत, निश्चित निदान नाहीत.",
    )
)

confirmed_tomato = st.checkbox(
    tr(
        "I confirm this is a tomato leaf",
        "मैं पुष्टि करता/करती हूँ कि यह टमाटर का पत्ता है",
        "हे टोमॅटोचे पान आहे याची मी पुष्टी करतो/करते",
    )
)

photo = st.file_uploader(
    tr(
        "Upload a tomato leaf photo",
        "टमाटर के पत्ते की फ़ोटो अपलोड करें",
        "टोमॅटोच्या पानाचा फोटो अपलोड करा",
    ),
    type=["jpg", "jpeg", "png"],
)

if photo:
    photo_bytes = photo.getvalue()
    photo_id = hashlib.sha256(photo_bytes).hexdigest()

    # Clear an earlier result when a different photo is uploaded.
    if st.session_state.get("photo_id") != photo_id:
        st.session_state["photo_id"] = photo_id
        st.session_state.pop("predictions", None)

    try:
        image = Image.open(BytesIO(photo_bytes)).convert("RGB")
        st.image(
            image,
            caption=tr(
                "Uploaded leaf",
                "अपलोड किया गया पत्ता",
                "अपलोड केलेले पान",
            ),
            use_container_width=True,
        )

        if st.button(
            tr("Check leaf", "पत्ते की जाँच करें", "पान तपासा"),
            disabled=not confirmed_tomato,
        ):
            try:
                with st.spinner(
                    tr(
                        "Looking at the leaf...",
                        "पत्ते की जाँच हो रही है...",
                        "पान तपासत आहे...",
                    )
                ):
                    st.session_state["predictions"] = load_classifier()(
                        image,
                        top_k=38,
                    )
            except Exception as error:
                st.error(
                    tr(
                        f"Could not load the classifier: {error}",
                        f"मॉडल लोड नहीं हो सका: {error}",
                        f"मॉडेल लोड करता आले नाही: {error}",
                    )
                )

        if not confirmed_tomato:
            st.info(
                tr(
                    "Confirm that the photo shows a tomato leaf to continue.",
                    "आगे बढ़ने के लिए पुष्टि करें कि फ़ोटो में टमाटर का पत्ता है।",
                    "पुढे जाण्यासाठी फोटोमध्ये टोमॅटोचे पान आहे याची पुष्टी करा.",
                )
            )

        predictions = st.session_state.get("predictions")

        if predictions and confirmed_tomato:
            top_overall = predictions[0]

            tomato_predictions = [
                item
                for item in predictions
                if item["label"].startswith("Tomato___")
            ]

            if not top_overall["label"].startswith("Tomato___"):
                st.warning(
                    tr(
                        "The model's strongest overall match is not tomato. "
                        "It may have mistaken the crop, so treat the result as uncertain.",
                        "मॉडल का सबसे मज़बूत मिलान टमाटर नहीं है। उसने फ़सल को "
                        "गलत पहचाना हो सकता है, इसलिए नतीजे को अनिश्चित मानें।",
                        "मॉडेलचा सर्वात मजबूत अंदाज टोमॅटोचा नाही. त्याने पीक "
                        "चुकीचे ओळखले असू शकते, म्हणून निष्कर्ष अनिश्चित समजा.",
                    )
                )
                st.caption(
                    tr(
                        "Strongest overall match: ",
                        "सबसे मज़बूत मिलान: ",
                        "सर्वात मजबूत अंदाज: ",
                    )
                    + f"{readable_label(top_overall['label'])} "
                    + f"({float(top_overall['score']):.0%})"
                )

            if not tomato_predictions:
                st.warning(
                    tr(
                        "The model found no tomato match. Try a clearer photo "
                        "of one tomato leaf.",
                        "मॉडल को टमाटर का मिलान नहीं मिला। टमाटर के एक पत्ते "
                        "की और साफ़ फ़ोटो लेकर देखें।",
                        "मॉडेलला टोमॅटोचा अंदाज मिळाला नाही. टोमॅटोच्या एका "
                        "पानाचा अधिक स्पष्ट फोटो वापरून पाहा.",
                    )
                )
            else:
                best_tomato = tomato_predictions[0]
                condition = best_tomato["label"].split("___", 1)[1]
                condition = condition.replace("_", " ")
                score = float(best_tomato["score"])

                st.subheader(
                    tr(
                        "Best tomato possibility",
                        "टमाटर की सबसे संभावित बीमारी",
                        "टोमॅटोचा सर्वाधिक संभाव्य रोग",
                    )
                )

                display_condition = condition
                if "early blight" in condition.lower():
                    display_condition = tr(
                        "Early blight",
                        "अगेती झुलसा (Early blight)",
                        "लवकर येणारा करपा (Early blight)",
                    )

                st.write(f"**{display_condition}**")
                st.caption(
                    f"{tr('Model score:', 'मॉडल स्कोर:', 'मॉडेल गुण:')} "
                    f"{score:.0%}"
                )

                if (
                    score < CONFIDENCE_THRESHOLD
                    or not top_overall["label"].startswith("Tomato___")
                ):
                    st.warning(
                        tr(
                            "The result is uncertain. Retake the photo or ask a "
                            "local agricultural expert before choosing a treatment.",
                            "नतीजा अनिश्चित है। उपचार चुनने से पहले दोबारा साफ़ "
                            "फ़ोटो लें या स्थानीय कृषि विशेषज्ञ से पूछें।",
                            "निष्कर्ष अनिश्चित आहे. उपचार ठरवण्यापूर्वी पुन्हा "
                            "स्पष्ट फोटो घ्या किंवा स्थानिक कृषी तज्ज्ञांचा सल्ला घ्या.",
                        )
                    )
                else:
                    st.info(
                        tr(
                            "This is a possible match, not a confirmed diagnosis.",
                            "यह एक संभावित मिलान है, पक्का निदान नहीं।",
                            "हा संभाव्य अंदाज आहे, निश्चित निदान नाही.",
                        )
                    )

                    if "early blight" in condition.lower():
                        st.markdown(
                            "### "
                            + tr(
                                "What early blight can look like",
                                "अगेती झुलसा कैसा दिख सकता है",
                                "लवकर येणारा करपा कसा दिसू शकतो",
                            )
                        )
                        st.write(
                            tr(
                                "It can start as small dark spots on older, lower "
                                "leaves. Larger spots may have ring-like patterns, "
                                "with yellowing around them.",
                                "यह पुराने, नीचे के पत्तों पर छोटे गहरे धब्बों से "
                                "शुरू हो सकता है। बड़े धब्बों में छल्ले जैसे निशान "
                                "और उनके आसपास पीलापन दिख सकता है।",
                                "हा रोग जुन्या, खालच्या पानांवर लहान गडद डागांनी "
                                "सुरू होऊ शकतो. मोठ्या डागांमध्ये वर्तुळाकार खुणा "
                                "आणि भोवती पिवळेपणा दिसू शकतो.",
                            )
                        )

                        st.markdown(
                            "### "
                            + tr(
                                "Lower-cost care steps",
                                "कम खर्च वाले देखभाल के कदम",
                                "कमी खर्चातील काळजीची पावले",
                            )
                        )
                        st.write(
                            tr(
                                "1. Check older, lower leaves for dark spots with "
                                "ring-like patterns.",
                                "1. पुराने, नीचे के पत्तों पर छल्ले जैसे निशान वाले "
                                "गहरे धब्बे देखें।",
                                "1. जुन्या, खालच्या पानांवर वर्तुळाकार खुणा असलेले "
                                "गडद डाग आहेत का ते पाहा.",
                            )
                        )
                        st.write(
                            tr(
                                "2. Water at the base of the plant and try to keep "
                                "leaves dry.",
                                "2. पौधे की जड़ के पास पानी दें और पत्तों को सूखा "
                                "रखने की कोशिश करें।",
                                "2. रोपाच्या बुंध्याशी पाणी द्या आणि पाने कोरडी "
                                "राहतील याची काळजी घ्या.",
                            )
                        )
                        st.write(
                            tr(
                                "3. Carefully remove spotted leaves. Clean your "
                                "tools afterward.",
                                "3. धब्बे वाले पत्तों को सावधानी से हटाएँ। बाद में "
                                "अपने औज़ार साफ़ करें।",
                                "3. डाग असलेली पाने काळजीपूर्वक काढा. नंतर "
                                "अवजारे स्वच्छ करा.",
                            )
                        )
                        st.write(
                            tr(
                                "4. Clear infected plant debris and reduce soil "
                                "splashing onto leaves.",
                                "4. संक्रमित पौधों के अवशेष हटाएँ और मिट्टी के "
                                "छींटे पत्तों पर पड़ने से रोकें।",
                                "4. रोगट अवशेष काढून टाका आणि मातीचे पाणी "
                                "पानांवर उडणार नाही याची काळजी घ्या.",
                            )
                        )
                        st.caption(
                            tr(
                                "If many leaves are affected, ask an agricultural "
                                "expert before pruning heavily.",
                                "अगर बहुत से पत्ते प्रभावित हैं, तो अधिक छँटाई "
                                "करने से पहले कृषि विशेषज्ञ से पूछें।",
                                "अनेक पाने बाधित असतील, तर जास्त छाटणी करण्यापूर्वी "
                                "कृषी तज्ज्ञांचा सल्ला घ्या.",
                            )
                        )

                        st.markdown(
                            "### "
                            + tr(
                                "If the problem continues",
                                "अगर समस्या बनी रहे",
                                "समस्या कायम राहिल्यास",
                            )
                        )
                        st.write(
                            tr(
                                "Show the photo and plant to a local KVK or "
                                "agricultural expert. Ask them to confirm the "
                                "disease and recommend treatment suitable for "
                                "your area and crop stage.",
                                "फ़ोटो और पौधे को नज़दीकी KVK या कृषि विशेषज्ञ "
                                "को दिखाएँ। उनसे बीमारी की पुष्टि करने और आपके "
                                "इलाके व फ़सल की अवस्था के अनुसार उपाय बताने को कहें।",
                                "फोटो आणि रोप जवळच्या KVK किंवा कृषी तज्ज्ञांना "
                                "दाखवा. रोगाची खात्री करून तुमच्या भागासाठी आणि "
                                "पिकाच्या अवस्थेसाठी योग्य उपाय विचारा.",
                            )
                        )
                        st.caption(
                            tr(
                                "This app does not provide pesticide doses. Use a "
                                "pesticide only if a local expert recommends it, "
                                "and follow the product label and harvest waiting period.",
                                "यह ऐप कीटनाशक की मात्रा नहीं बताता। कीटनाशक का "
                                "उपयोग केवल स्थानीय विशेषज्ञ की सलाह पर करें और "
                                "उत्पाद के लेबल तथा कटाई से पहले की प्रतीक्षा अवधि "
                                "का पालन करें।",
                                "हे अॅप कीटकनाशकाचे प्रमाण सांगत नाही. स्थानिक "
                                "तज्ज्ञांनी सुचवल्यासच कीटकनाशक वापरा आणि "
                                "उत्पादनाच्या लेबलवरील व काढणीपूर्व प्रतीक्षा "
                                "कालावधीच्या सूचनांचे पालन करा.",
                            )
                        )

                        st.markdown(
                            "### "
                            + tr("Sources", "स्रोत", "स्रोत")
                        )
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
                            tr(
                                "Simple, source-checked guidance for this possible "
                                "condition has not been added yet. Ask a local "
                                "agricultural expert.",
                                "इस संभावित समस्या के लिए स्रोतों से जाँची गई "
                                "सरल सलाह अभी उपलब्ध नहीं है। स्थानीय कृषि "
                                "विशेषज्ञ से पूछें।",
                                "या संभाव्य समस्येसाठी स्रोत तपासून तयार केलेले "
                                "सोपे मार्गदर्शन अजून उपलब्ध नाही. स्थानिक कृषी "
                                "तज्ज्ञांचा सल्ला घ्या.",
                            )
                        )

            st.subheader(
                tr("Why this result?", "यह नतीजा क्यों?", "हा निष्कर्ष का?")
            )
            st.write(
                tr(
                    "The model compared the photo with its trained crop and "
                    "disease labels.",
                    "मॉडल ने फ़ोटो की तुलना उन फ़सल और बीमारी के नामों से की "
                    "जिन पर उसे प्रशिक्षित किया गया है।",
                    "मॉडेलने फोटोची तुलना ज्या पीक आणि रोगांच्या नावांवर "
                    "प्रशिक्षण दिले आहे त्यांच्याशी केली.",
                )
            )

            with st.expander(
                tr(
                    "See the model's top three overall matches",
                    "मॉडल के तीन सबसे मज़बूत मिलान देखें",
                    "मॉडेलचे तीन सर्वात मजबूत अंदाज पहा",
                )
            ):
                for item in predictions[:3]:
                    st.write(
                        f"{readable_label(item['label'])}: "
                        f"{float(item['score']):.0%}"
                    )

            st.caption(
                tr(
                    "These scores are model outputs, not verified probabilities. "
                    "Crop and disease labels in the match list are shown in the "
                    "model's original English.",
                    "ये स्कोर मॉडल के नतीजे हैं, सत्यापित संभावनाएँ नहीं। "
                    "मिलान सूची में फ़सल और बीमारी के नाम मॉडल की मूल अंग्रेज़ी "
                    "में दिखाए गए हैं।",
                    "हे गुण मॉडेलचे निष्कर्ष आहेत, पडताळलेल्या शक्यता नाहीत. "
                    "अंदाज यादीतील पीक आणि रोगांची नावे मॉडेलच्या मूळ इंग्रजीत "
                    "दाखवली आहेत.",
                )
            )

            st.subheader(
                tr("Farmer observations", "किसान की जानकारी", "शेतकऱ्याची माहिती")
            )
            st.write(
                tr(
                    "These answers add context. They do not change the model result.",
                    "इन जवाबों से संदर्भ मिलता है। इनसे मॉडल का नतीजा नहीं बदलता।",
                    "या उत्तरांमुळे संदर्भ मिळतो. यामुळे मॉडेलचा निष्कर्ष बदलत नाही.",
                )
            )

            symptoms = st.multiselect(
                tr(
                    "What do you notice on the leaf?",
                    "पत्ते पर क्या दिखाई देता है?",
                    "पानावर काय दिसते?",
                ),
                [
                    tr("Dark or brown spots", "गहरे या भूरे धब्बे", "गडद किंवा तपकिरी डाग"),
                    tr("Yellowing", "पीलापन", "पिवळेपणा"),
                    tr("Curling", "मुड़ना", "पाने वळणे"),
                    tr("Dry or damaged edges", "सूखे या खराब किनारे", "कोरड्या किंवा खराब कडा"),
                    tr("Powdery-looking patches", "पाउडर जैसे धब्बे", "पावडरसारखे डाग"),
                    tr("Not sure", "पता नहीं", "खात्री नाही"),
                ],
            )

            watering = st.selectbox(
                tr(
                    "How has watering been recently?",
                    "हाल में पानी कितना दिया गया है?",
                    "अलीकडे पाणी किती दिले आहे?",
                ),
                [
                    tr("About usual", "सामान्य जितना", "नेहमीइतके"),
                    tr("Less than usual", "सामान्य से कम", "नेहमीपेक्षा कमी"),
                    tr("More than usual", "सामान्य से ज़्यादा", "नेहमीपेक्षा जास्त"),
                    tr("Soil is often waterlogged", "मिट्टी में अक्सर पानी भरा रहता है", "मातीमध्ये वारंवार पाणी साचते"),
                    tr("Not sure", "पता नहीं", "खात्री नाही"),
                ],
            )

            duration = st.selectbox(
                tr(
                    "When did you first notice the change?",
                    "यह बदलाव पहली बार कब दिखा?",
                    "हा बदल पहिल्यांदा कधी दिसला?",
                ),
                [
                    tr("Today", "आज", "आज"),
                    tr("In the last few days", "पिछले कुछ दिनों में", "गेल्या काही दिवसांत"),
                    tr("More than a week ago", "एक हफ़्ते से ज़्यादा पहले", "एका आठवड्यापेक्षा जास्त आधी"),
                    tr("Not sure", "पता नहीं", "खात्री नाही"),
                ],
            )

            st.write(f"**{tr('Your notes:', 'आपकी जानकारी:', 'तुमची नोंद:')}**")
            st.write(
                f"- {tr('Leaf signs:', 'पत्ते के लक्षण:', 'पानांची लक्षणे:')} "
                f"{', '.join(symptoms) if symptoms else tr('Not answered', 'जवाब नहीं दिया', 'उत्तर दिले नाही')}"
            )
            st.write(f"- {tr('Watering:', 'पानी:', 'पाणी:')} {watering}")
            st.write(f"- {tr('Started:', 'शुरुआत:', 'सुरुवात:')} {duration}")

            st.caption(
                tr(
                    "This prototype does not save these answers after you close "
                    "the app. The observations do not affect the model's result.",
                    "ऐप बंद करने के बाद यह नमूना आपके जवाब सहेजता नहीं है। "
                    "ये जानकारी मॉडल के नतीजे को प्रभावित नहीं करती।",
                    "अॅप बंद केल्यानंतर हा नमुना तुमची उत्तरे साठवत नाही. "
                    "या नोंदींचा मॉडेलच्या निष्कर्षावर परिणाम होत नाही.",
                )
            )

    except Exception as error:
        st.error(
            tr(
                f"Could not open the uploaded photo: {error}",
                f"अपलोड की गई फ़ोटो नहीं खुल सकी: {error}",
                f"अपलोड केलेला फोटो उघडता आला नाही: {error}",
            )
        )