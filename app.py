import streamlit as st
import joblib
from pathlib import Path

# Step 1: Configure Website
st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️"
)

st.title(" 🛡️ AI Phishing Email Detector")

st.write(
    "Enter an email message below to check whether it might be a phishing attempt"
)

# Step 2: Load trained AI model
model_path = Path(__file__).with_name("phishing_model.joblib")
model = joblib.load(model_path)


# STEP 3: Provide example emails for the demo

sample_emails = {
    "Custom Email": "",

    "Phishing - Prize Scam":
        "I am going to give you 1 bajillion dollars. "
        "Click the link below: bit.ly/1234141",

    "Legitimate - Project Meeting":
        "The project meeting is scheduled for tomorrow morning",

    "Phishing - Gift Card Scam":
        "Finance department urgently needs gift cards "
        "purchased and the codes emailed back",

    "Legitimate - Security Reminder":
        "Never share your password or verification "
        "code with anyone"
}

# Expected labels for our predefined demo examples
# 1 = Phishing, 0 = Legitimate

sample_labels = {
    "Phishing - Prize Scam": 1,
    "Legitimate - Project Meeting": 0,
    "Phishing - Gift Card Scam": 1,
    "Legitimate - Security Reminder": 0
}


# Create a dropdown menu
selected_example = st.selectbox(
    "Choose a sample email:",
    list(sample_emails.keys())
)

# Display the selected email or allow custom input
email_text = st.text_area(
    "Enter email content:",
    value=sample_emails[selected_example],
    height=180,
    placeholder="Paste your email here..."
)

# Step 4: Create an Analyze button
if st.button("Analyze Email"):

    if not email_text.strip():
        st.warning("Please enter an email first")

    else:
        # Step 5: Predicting using AI model
        prediction = model.predict([email_text])[0]

        phishing_score = model.predict_proba(
            [email_text]
        )[0][1]

        # Step 6 display results
        st.subheader("Analysis Result")

        if prediction == 1:
            st.error("⚠️ Potential Phishing Email")

        else:
            st.success("✅ Likely Legitimate Email")

        st.write(
            f"Model phishing score: {phishing_score:.2%}"
        )

        st.progress(int(phishing_score * 100))


        # Compare model prediction with the expected demo label
        if (
            selected_example in sample_labels
            and email_text.strip() == sample_emails[selected_example].strip()
        ):

            expected = sample_labels[selected_example]

            expected_text = (
                "Phishing" if expected == 1 else "Legitimate"
            )

            st.write(f"**Expected classification:** {expected_text}")

            if prediction == expected:
                st.success("Model correctly classified this example.")
            else:
                st.warning(
                    "Model misclassified this example. "
                    "This demonstrates a false positive "
                    "or false negative."
                )


        # STEP 7: Explain the AI prediction

        # Access the two components of our trained model
        tfidf = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        # Convert the entered email into numerical features
        vector = tfidf.transform([email_text])

        # Get the learned weights and feature names
        weights = classifier.coef_[0]
        feature_names = tfidf.get_feature_names_out()

        # Calculate how much each word contributes
        contributions = vector.multiply(weights).toarray()[0]

        # Find the 5 most influential features
        top_indices = abs(contributions).argsort()[::-1][:5]

        # Display explanations in an expandable section
        with st.expander("🔍 Why did the AI make this prediction?"):

            found_features = False

            for i in top_indices:
                if contributions[i] == 0:
                    continue

                found_features = True
                word = feature_names[i]

                if contributions[i] > 0:
                    st.write(f"🔴 '{word}' pushes toward phishing")
                else:
                    st.write(f"🟢 '{word}' pushes toward legitimate")

            if not found_features:
                st.info("No recognized training vocabulary words found.")


        st.caption(
            "This is a demonstration model trained "
            "on a small synthetic dataset. "
            "Its prediction is not a security guarantee."
        )














