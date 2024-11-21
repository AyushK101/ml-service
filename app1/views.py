from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import joblib
import os
import random
from dotenv import load_dotenv




# Create your views here.
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')


equipment_data = {
    "Refrigerator": {"power": 150, "usage_hours": 24},
    "Air Conditioner": {"power": 2000, "usage_hours": 8},
    "Washing Machine": {"power": 500, "usage_hours": 1},
    "Lighting": {"power": 10, "usage_hours": 5},
    "TV": {"power": 120, "usage_hours": 4},
    "Microwave": {"power": 800, "usage_hours": 0.5},
    "Fan": {"power": 75, "usage_hours": 12},
    "Computer": {"power": 300, "usage_hours": 6},
}

# Define solar irradiance data by city (in kWh/m²/day)
location_irradiance = {
    "mumbai": 4.5,
    "delhi": 5.2,
    "chennai": 4.8,
    "bangalore": 5.0,
    "kolkata": 4.6,
    "pune": 5.1,
    "bhopal": 4.9,
    "indore": 5.0
}


def health(request):
  return HttpResponse("health check successful 🟢")


def calculate_solar_capacity(appliances, city):
    """
    Calculates the required solar panel capacity for a list of appliances in a given city.

    Parameters:
        appliances (list of dicts): List where each dict has 'name' (appliance), 'quantity', 'power', 'usage_hours'.
        city (str): City name to use the appropriate solar irradiance value.

    Returns:
        float: Required solar panel capacity in kW.
    """
    # Validate city
    if str(city).lower() not in location_irradiance:
        raise ValueError(
            f"City '{str(city).lower()}' is not available. Choose from: {', '.join(location_irradiance.keys())}")

    # Get the solar irradiance for the specified city
    irradiance = location_irradiance[str(city).lower()]

    # Calculate total daily energy requirement in kWh
    total_energy_kwh = 0
    for appliance in appliances:
        name = appliance["appliance"]
        quantity = appliance["quantity"]

        # Retrieve power and usage hours from default data if not provided
        power = appliance.get("power", equipment_data.get(name, {}).get("power"))
        usage_hours = appliance.get("usageHours", equipment_data.get(name, {}).get("usageHours"))

        if power is None or usage_hours is None:
            raise ValueError(f"Details for appliance '{name}' are missing or not found.")

        # Calculate energy for this appliance and add to total
        daily_energy = (power * usage_hours * quantity) / 1000  # Convert W-hours to kWh
        total_energy_kwh += daily_energy

    # Calculate required solar capacity (kW) based on irradiance
    required_solar_kw = total_energy_kwh / irradiance
    return round(required_solar_kw, 2)

@csrf_exempt
def calculate_solar(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        print("data",data)
        city = str(data.get('city'))
        print("city",city)
        appliances = data.get('applianceList', [])
        print(appliances)
        # print(appliances)

        solar_requirement_kw = calculate_solar_capacity(city=city, appliances=appliances)
        # solar_requirement_kw = 55

        return JsonResponse({
            'city': city,
            'solar_requirement': solar_requirement_kw
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)


def predict_loads(X1, X2, X3, X4, X5, X6, X7, X8):
    best_model = load_model('model/best_model.keras')
    scaler = joblib.load('model/scaler.pkl')
    X_new = np.array([[X1, X2, X3, X4, X5, X6, X7, X8]])
    X_new_scaled = scaler.transform(X_new)
    y1_pred, y2_pred = best_model.predict(X_new_scaled)
    heating_load = y1_pred[0][0]
    cooling_load = y2_pred[0][0]
    return heating_load, cooling_load


@csrf_exempt
def predict_view(request):
    print("inside predict_view")
    if request.method == 'POST':
        # Get form data
        data = json.loads(request.body.decode('utf-8'))
        print(data["relativeCompactNess"])
        X1 = float(data["relativeCompactNess"])
        X2 = float(data["surfaceArea"])
        X3 = float(data["wallArea"])
        X4 = float(data["roofArea"])
        X5 = float(data["overallHeight"])
        X6 = float(data["orientation"])
        X7 = float(data["glazingArea"])
        X8 = float(data["distribution"])

        # Call the prediction function
        heating_load, cooling_load = predict_loads(X1, X2, X3, X4, X5, X6, X7, X8)

        heating_load = float(heating_load)
        cooling_load = float(cooling_load)

        # Example logic to determine device usage times based on load
        fan_hours = cooling_load / 100  # Sample calculation for fan time
        ac_hours = cooling_load / 500   # Sample calculation for AC time

        # Return the result as a JSON response (to trigger modal with prediction)
        return JsonResponse({
            'heating_load': round(heating_load, 2),
            'cooling_load': round(cooling_load, 2),
            'fan_hours': round(fan_hours*24, 2),
            'ac_hours': round(ac_hours*24, 2)
        })
    

genai.configure(api_key=API_KEY)



@csrf_exempt
def gen(request):
    message = ""
    bot_response = ""

    if request.method == "POST":
        message =  json.loads(request.body)
        # return JsonResponse({"message":message})
        if message:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(f"I know you are not expert just tell me genric answer in just 2 to 3 lines and "
                                                  f"be specific about the answer do not go here and there just say answer nothing else "
                                                  f"dont be in doubt just say one answer i know that answer maybe be "
                                                  f"incorrect but just say answer do not say impossible or hard to say just give me one answer "
                                                  f"when asked about government schemes, give links of government websites for specific"
                                                  f"the question is by User: {message}")
                bot_response = response.text if response else "I'm sorry, I couldn't understand that."

            except Exception as e:
                print("Error generating response:", e)
                bot_response = "There was an error generating a response. Please try again."

    return JsonResponse({"message": message, "response": bot_response})