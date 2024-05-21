import pandas as pd

# CSV-Datei laden
file_path = 'apartments_data_zurich_30.12.2023.csv'  # Ersetzen Sie dies mit dem Pfad zu Ihrer CSV-Datei
data = pd.read_csv(file_path)

# Daten neu formatieren
formatted_data = pd.DataFrame({
    "Rooms": data['rooms_area_price_raw'].str.extract(r'(\d+,\d+|\d+) Zimmer')[0],
    "Area": data['rooms_area_price_raw'].str.extract(r'(\d+,\d+|\d+) m²')[0],
    "Price": data['price_raw'].str.replace('CHF', '').str.replace('.—', '').str.replace('PreisaufAnfrage', '').str.replace(' ', '').str.strip(),  # CHF, .— und 'PreisaufAnfrage' entfernen, Leerzeichen entfernen und trimmen
    "City": data['address_raw'].str.extract(r'(\d{4} \w+), ZH')[0].str.split(n=1).str[1]  # Postleitzahl entfernen
})

# Leere Werte in den kritischen Spalten entfernen
formatted_data.dropna(subset=['Rooms', 'Area', 'Price', 'City'], how='any', inplace=True)

# Preis auf Anfrage entfernen und andere nicht-numerische Einträge
formatted_data = formatted_data[~formatted_data['Price'].str.contains('PreisaufAnfrage', na=False)]

# "(m²,Jahr)" aus der Price-Spalte entfernen
formatted_data['Price'] = formatted_data['Price'].str.replace('(m²,Jahr)', '')

# Optional: Konvertierung von Komma in Punkte für numerische Werte, falls erforderlich
formatted_data['Rooms'] = formatted_data['Rooms'].str.replace(',', '.').astype(float)
formatted_data['Area'] = formatted_data['Area'].str.replace(',', '.').astype(float)
formatted_data['Price'] = formatted_data['Price'].str.replace(',', '.').astype(float)

# Ausgabe in neuer CSV-Datei speichern
formatted_data.to_csv('cleaned_rent_data.csv', index=False)

# Erste Zeilen zur Überprüfung anzeigen
print(formatted_data.head())