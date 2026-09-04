import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import pickle
import base64
import plotly.express as px


# ---------------- BACKGROUND FUNCTION ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Price Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("mlr.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- LOAD DATA FOR VISUALS ----------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("cardekho_imputated.csv")
    except:
        return None

cars_df = load_data()

# ---------------- PREDICT FUNCTION ----------------
def predict_price(vehicle_age, km_driven, seller_type, fuel_type,
                  transmission_type, mileage, engine, max_power, seats):

    x = np.array([[vehicle_age, km_driven, seller_type, fuel_type,
                   transmission_type, mileage, engine, max_power, seats]])
    return float(model.predict(x)[0])

# ---------------- CAR IMAGE DECISION LOGIC ----------------
def get_car_image(fuel_type, transmission, engine, power, seats):
    """
    Select the car image based on ALL features.
    Priority → Electric > Sports > SUV > Sedan > Hatchback > Transmission-based.
    """

    # Priority 1: Electric cars
    if fuel_type == 4:
        return "electric.jpg"

    # Priority 2: Sports / High performance
    if power > 120 or engine > 1800:
        return "sports.jpg"

    # Priority 3: SUV (7 or more seats)
    if seats >= 7:
        return "suv.jpg"

    # Priority 4: Sedan (5 seats)
    if seats == 5:
        return "sedan.jpg"

    # Priority 5: Hatchback (small engine)
    if engine < 1200:
        return "hatchback.jpg"

    # Priority 6: Based on transmission
    if transmission == 1:
        return "automatic.jpg"
    else:
        return "manual.jpg"

    # Fallback
    return "default.jpg"

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #000000 !important;   /* Light background */
    padding: 25px;
    color: #1A1A1A !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4, 
[data-testid="stSidebar"] p {
    color: #FFFFFF !important;   /* Dark text */
}

/* Menu text color */
.nav-link {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Navigation")
    

    selected = option_menu(
        menu_title="",
        options=["Home", "Predict", "Dataset Overview", "Author"],
        icons=["house", "speedometer2", "bar-chart", "person-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "nav-link": {"font-size": "18px", "color": "#1A1A1A"},
            "nav-link-selected": {"background-color": "#FF6666", "color": "white", "border-radius": "10px"},
        }
)


# --------------- HOME PAGE ----------------
if selected == "Home":
    set_bg("home_bg.jpg")

    st.markdown(
        """
<div style="background: rgba(0,0,0,0.55); padding: 25px; border-radius: 12px;
            width: 80%; margin: auto; margin-top: 30px;">
  <h2 style="color: white; text-align: center; font-family: Poppins, sans-serif;">
    How the Model Works
  </h2>

  <ul style="color: #E8E8E8; font-size: 18px; line-height: 1.7; font-family: Poppins, sans-serif;">
    <li>Collects historical car data (price, mileage, engine, power, age, etc.)</li>
    <li>Converts categorical fields into encoded values</li>
    <li>Processes numerical features with regression</li>
    <li>Predicts the best possible selling price for the car</li>
    <li>Displays a matching animated car visual for your inputs</li>
  </ul>
</div>
        """,
        unsafe_allow_html=True
    )

# --------------- PREDICT PAGE ----------------
elif selected == "Predict":
    st.header("Predict Car Price")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter Car Details")

    # numerical inputs
    vehicle_age = st.number_input("Vehicle Age (years)", 0, 30, 5)
    km_driven = st.number_input("Kilometers Driven", 0, 300000, 45000)
    mileage = st.number_input("Mileage (kmpl)", 0.0, 50.0, 19.5)
    engine = st.number_input("Engine CC", 500, 5000, 1200)
    max_power = st.number_input("Max Power (bhp)", 20.0, 300.0, 82.0)
    seats = st.number_input("Seats", 2, 10, 5)

    st.markdown("### Encoded Categorical Inputs")
    seller_type = st.number_input("Seller Type (encoded)", 0, 2, 1)
    fuel_type = st.number_input("Fuel Type (encoded)", 0, 4, 0)
    transmission_type = st.number_input("Transmission Type (encoded)", 0, 1, 0)

    if st.button("Predict"):
        price = predict_price(
            vehicle_age, km_driven, seller_type, fuel_type,
            transmission_type, mileage, engine, max_power, seats
        )

        st.success(f"Predicted Price: ₹ {price:,.0f}")

        # ---------------- SHOW CAR IMAGE ----------------
        img_name = get_car_image(fuel_type, transmission_type, engine, max_power, seats)
        img_path = f"car_images/{img_name}"
        st.image(img_path, caption="Car type based on your inputs", width=450)

    st.markdown("</div>", unsafe_allow_html=True)
# ---------------- DATASET OVERVIEW PAGE ----------------
# ---------------- ADVANCED DATASET OVERVIEW PAGE ----------------
elif selected == "Dataset Overview":

    st.title("📊 Advanced Dataset Overview")

    if cars_df is None:
        st.warning("Dataset not found!")
    else:

        # ===============================================================
        #  FILTERS
        # ===============================================================
        st.sidebar.subheader("🔎 Filters")

        # Brand Filter (if available)
        if "brand" in cars_df.columns:
            all_brands = ["All"] + sorted(cars_df["brand"].unique().tolist())
            selected_brand = st.sidebar.selectbox("Select Brand", all_brands)
        else:
            selected_brand = "All"

        # Fuel Filter
        if "fuel_type" in cars_df.columns:
            all_fuels = ["All"] + sorted(cars_df["fuel_type"].unique().tolist())
            selected_fuel = st.sidebar.selectbox("Select Fuel Type", all_fuels)
        else:
            selected_fuel = "All"

        # Year/Age Filter
        if "vehicle_age" in cars_df.columns:
            min_age = int(cars_df["vehicle_age"].min())
            max_age = int(cars_df["vehicle_age"].max())
            selected_age = st.sidebar.slider("Vehicle Age Range", min_age, max_age, (min_age, max_age))
        else:
            selected_age = None

        # Apply filters
        filtered_df = cars_df.copy()

        if selected_brand != "All" and "brand" in cars_df.columns:
            filtered_df = filtered_df[filtered_df["brand"] == selected_brand]

        if selected_fuel != "All" and "fuel_type" in cars_df.columns:
            filtered_df = filtered_df[filtered_df["fuel_type"] == selected_fuel]

        if selected_age and "vehicle_age" in cars_df.columns:
            filtered_df = filtered_df[
                (filtered_df["vehicle_age"] >= selected_age[0]) &
                (filtered_df["vehicle_age"] <= selected_age[1])
            ]

        st.markdown("---")

        # ===============================================================
        #  TABS FOR INTERACTION
        # ===============================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Summary", 
            "📊 Charts", 
            "🔥 Correlation Heatmap", 
            "🚗 Brand-wise Pricing",
            "📈 Trendline Regression"
        ])

        # ===============================================================
        #  TAB 1 — SUMMARY
        # ===============================================================
        with tab1:
            st.subheader("Dataset Summary")
            st.write(f"**Rows:** {filtered_df.shape[0]}")
            st.write(f"**Columns:** {filtered_df.shape[1]}")
            st.dataframe(filtered_df.head())

        # ===============================================================
        #  TAB 2 — CHARTS
        # ===============================================================
        with tab2:
            import plotly.express as px

            # PRICE VS AGE
            if "vehicle_age" in filtered_df.columns and "selling_price" in filtered_df.columns:
                fig1 = px.scatter(
                    filtered_df, x="vehicle_age", y="selling_price",
                    color="fuel_type" if "fuel_type" in filtered_df.columns else None,
                    trendline="ols", title="Price vs Vehicle Age"
                )
                st.plotly_chart(fig1)

            # PRICE VS KM DRIVEN
            if "km_driven" in filtered_df.columns and "selling_price" in filtered_df.columns:
                fig2 = px.scatter(
                    filtered_df, x="km_driven", y="selling_price",
                    color="fuel_type" if "fuel_type" in filtered_df.columns else None,
                    trendline="ols", title="Price vs KM Driven"
                )
                st.plotly_chart(fig2)

            # FUEL DISTRIBUTION
            if "fuel_type" in filtered_df.columns:
                fig3 = px.histogram(
                    filtered_df, x="fuel_type", color="fuel_type",
                    title="Fuel Type Distribution"
                )
                st.plotly_chart(fig3)

            # ENGINE VS MAX POWER
            if "engine" in filtered_df.columns and "max_power" in filtered_df.columns:
                fig4 = px.scatter(
                    filtered_df, x="engine", y="max_power",
                    color="fuel_type" if "fuel_type" in filtered_df.columns else None,
                    title="Engine CC vs Max Power"
                )
                st.plotly_chart(fig4)

        # ===============================================================
        #  TAB 3 — CORRELATION HEATMAP
        # ===============================================================
        with tab3:
            import seaborn as sns
            import matplotlib.pyplot as plt

            st.subheader("Correlation Heatmap")

            # Only numeric columns
            numeric_df = filtered_df.select_dtypes(include=['int64', 'float64'])

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)

        # ===============================================================
        #  TAB 4 — BRAND-WISE PRICING
        # ===============================================================
        with tab4:
            if "brand" in filtered_df.columns and "selling_price" in filtered_df.columns:
                brand_prices = filtered_df.groupby("brand")["selling_price"].mean().sort_values(ascending=False)

                fig5 = px.bar(
                    brand_prices, 
                    title="Average Selling Price by Brand",
                    labels={"value": "Avg Price", "brand": "Brand"}
                )
                st.plotly_chart(fig5)
            else:
                st.info("Brand or selling_price column missing.")

        # ===============================================================
        #  TAB 5 — REGRESSION TRENDLINES
        # ===============================================================
        with tab5:
            st.subheader("Regression Trendline Insights")

            if "vehicle_age" in filtered_df.columns and "selling_price" in filtered_df.columns:
                fig6 = px.scatter(
                    filtered_df, x="vehicle_age", y="selling_price",
                    trendline="ols", title="Regression: Age → Price"
                )
                st.plotly_chart(fig6)

            if "engine" in filtered_df.columns and "selling_price" in filtered_df.columns:
                fig7 = px.scatter(
                    filtered_df, x="engine", y="selling_price",
                    trendline="ols", title="Regression: Engine → Price"
                )
                st.plotly_chart(fig7)







# --------------- AUTHOR PAGE ----------------
elif selected == "Author":
    st.title("About the Author")

    st.markdown(
        """
<div style="background: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 12px;
            width: 80%; margin: auto; margin-top: 20px;">
  <h2 style="color: #333; text-align: center;">Pradeep CS</h2>

  <p style="color: #444; font-size: 18px; text-align: center;">
    Data Science Enthusiast<br>
    Specializing in Machine Learning & Predictive Analytics<br>
    Creator of this Car Price Prediction Dashboard.
  </p>
</div>
        """,
        unsafe_allow_html=True
    )

