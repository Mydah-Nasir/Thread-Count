import streamlit as st
import base64
import json
from PIL import Image
import io
from openai import OpenAI
import re
from collections import defaultdict
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Industrial Spool Counter",
    page_icon="🧵",
    layout="wide"
)

# Initialize OpenAI
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please set your OpenAI API key in .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- HELPER FUNCTIONS ----------------
def image_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def safe_json_parse(content: str):
    # Extracts JSON even if GPT includes conversational prose
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}

def extract_spool_data(image: Image.Image):
    image_base64 = image_to_base64(image)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a specialized industrial OCR system for thread spool inventory.\n\n"
                    "DETECTION RULES:\n"
                    "1. Scan the image for EVERY spool. Count it even if the label is dark or tilted.\n"
                    "2. For each label, look specifically for the 'Shade:' field. Ignore 'Lot no' and 'Tkt'.\n"
                    "3. Text may be rotated or upside down; read it carefully.\n"
                    "4. If a spool is clearly present but you cannot read the shade, label it as 'Unreadable'.\n\n"
                    "Return ONLY JSON:\n"
                    "{\n"
                    "  \"spools\": [\n"
                    "    {\"row\": 1, \"col\": 1, \"shade\": \"20194 (984)\", \"status\": \"Detected\"}\n"
                    "  ]\n"
                    "}"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Perform a full inventory count. Identify every spool and extract the Shade number only. Disregard other numbers like Lot or Tkt."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high" 
                        }
                    }
                ]
            }
        ],
    )

    return safe_json_parse(response.choices[0].message.content)

# ---------------- UI ----------------
st.title("🧵 Industrial Spool Scanner")
st.markdown("Automated grid counting and shade extraction for warehouse inventory.")

uploaded_file = st.file_uploader("Upload Grid or Close-up Image", type=["jpg", "jpeg", "png"])
print('file uploaded')
if uploaded_file:
    # 2026 Layout style
    col_img, col_data = st.columns([1, 1])
    
    image = Image.open(uploaded_file)
    
    with col_img:
        # Using 2026 'width' parameter instead of deprecated 'use_container_width'
        st.image(image, caption="Current View", width='stretch')

    if st.button("Scan Inventory", type="primary"):
        with st.spinner("Processing image with High-Detail OCR..."):
            result = extract_spool_data(image)
        
        spools = result.get("spools", [])
        total_found = len(spools)

        if total_found == 0:
            st.error("No spools detected. Ensure the labels are visible and try again.")
        else:
            with col_data:
                st.metric("Total Count", total_found)
                
                # Process data for summary
                df = pd.DataFrame(spools)
                
                # Show results table
                st.subheader("Audit Log")
                st.dataframe(df, width='stretch')

                # Export option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Export CSV", csv, "inventory_audit.csv", "text/csv")

            st.divider()
            
            # Bottom Summary Cards
            st.subheader("Distribution by Shade")
            shade_counts = defaultdict(int)
            for s in spools:
                shade_counts[s['shade']] += 1
            
            stat_cols = st.columns(min(4, len(shade_counts) if len(shade_counts) > 0 else 1))
            for i, (shade, count) in enumerate(sorted(shade_counts.items())):
                with stat_cols[i % 4]:
                    st.metric(label=f"Shade {shade}", value=f"{count} qty")

else:
    st.info("Please upload an image to begin scanning.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Side Nav")
    st.divider()
    st.caption("Tip: If the count is wrong, ensure there is no direct glare on the white labels.")