use geojson::{Feature, FeatureCollection};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Station {
    pub station_long_name: String,
    pub station_name: String,
    pub station_id: String,
    pub latitude: String,
    pub longitude: String,
    pub elevation: String,
    pub station_fault: Option<String>,
    pub observation_time: Option<String>,
    pub timezone: Option<String>,
    pub temperature: Option<String>,
    pub temperature_low: Option<String>,
    pub temperature_high: Option<String>,
    pub temperature_units: Option<String>,
    pub temperature_inside: Option<String>,
    pub temperature_inside_units: Option<String>,
    pub humidity: Option<String>,
    pub humidity_units: Option<String>,
    pub dewpoint: Option<String>,
    pub dewpoint_units: Option<String>,
    pub wetbulb: Option<String>,
    pub wetbulb_units: Option<String>,
    pub pressure: Option<String>,
    pub pressure_units: Option<String>,
    pub pressure_trend: Option<String>,
    pub insolation: Option<String>,
    pub insolation_units: Option<String>,
    pub uv_index: Option<String>,
    pub uv_index_units: Option<String>,
    pub rain: Option<String>,
    pub rain_units: Option<String>,
    pub rain_rate: Option<String>,
    pub rain_rate_units: Option<String>,
    pub wind_speed: Option<String>,
    pub wind_speed_direction: Option<String>,
    pub wind_speed_heading: Option<String>,
    pub wind_speed_direction_units: Option<String>,
    pub wind_speed_max: Option<String>,
    pub wind_speed_units: Option<String>,
    pub insolation_predicted: Option<String>,
    pub insolation_predicted_units: Option<String>,
    pub error_message: Option<String>,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct ConditionsWrapper {
    pub current_conditions: Conditions,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Conditions {
    #[serde(rename = "$")]
    metadata: serde_json::Value,
    credit: String,
    credit_url: String,
    disclaimer: String,
    description: String,
    pub current_observation: Vec<Station>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Intersection {
    pub intersection: FeatureCollection,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Points {
    pub points: FeatureCollection,
    pub average_temp: f64,
    pub max_point: Option<Feature>,
    pub min_point: Option<Feature>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Rain {
    pub max_rain: Option<Feature>,
    pub average_rain: f64,
    pub number_reporting: i64,
}
