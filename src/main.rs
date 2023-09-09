pub mod types;
use crate::types::ConditionsWrapper;
use anyhow::{Error, Result};
use axiom_rs;
use contour::{Contour, ContourBuilder};
use dotenvy::dotenv;
use friedrich::{gaussian_process::GaussianProcess, kernel::Exponential};
use geo::{BooleanOps, BoundingRect, Coord, LineString, MultiPolygon, Point, Polygon};
use geojson::{Feature, FeatureCollection, Value};
use iter_num_tools::{arange_grid, lin_space};
use mongodb::{bson::doc, Client};
use serde_json::json;
use tokio::time::{self, Duration, Instant};
use types::{Intersection, Points, Rain, Station};
use xml2json_rs::JsonConfig;

const NUMBER_OF_BANDS: usize = 15;
const COORDINATE_STEP_X: f64 = 0.01;
const COORDINATE_STEP_Y: f64 = 0.01;
const MODEL_NOISE: f64 = 0.01;
const MAX_LAT: f64 = 60.0;
const WAIT_TIME: u64 = 60;

pub struct Logger {
    client: axiom_rs::Client,
}

impl Logger {
    pub fn new() -> Self {
        match axiom_rs::Client::new() {
            Ok(client) => Self { client },
            Err(e) => {
                panic!("Failed to create axiom client: {}", e);
            }
        }
    }
    pub async fn log(&self, message: &Error) {
        match self
            .client
            .datasets
            .ingest(
                "vercel",
                vec![json!({
                    "message": message.to_string(),
                    "timestamp": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                })],
            )
            .await
        {
            Ok(_) => {
                println!("Logged message: {}", message);
            }
            Err(e) => {
                println!("Failed to log message: {}", e);
            }
        }
    }
}

struct Storage {
    database: mongodb::Database,
}

impl Storage {
    pub async fn new() -> Self {
        let user = std::env::var("MONGO_USER").unwrap();
        let password = std::env::var("MONGO_PASS").unwrap();

        let uri = format!(
            "mongodb+srv://{}:{}@victoria-weather.hzivz.mongodb.net/?retryWrites=true&w=majority",
            user, password
        );

        match Client::with_uri_str(uri).await {
            Ok(client) => {
                let database = client.database("weather-test");
                Self { database }
            }
            Err(e) => {
                panic!("Failed to get mongo client: {}", e);
            }
        }
    }
    pub async fn store_intersection(&self, data: Intersection) -> Result<()> {
        let collection = self.database.collection::<Intersection>("intersection");
        let inserted = collection.insert_one(data, None).await?;
        collection
            .delete_many(doc! {"_id": { "$ne": inserted.inserted_id }}, None)
            .await?;

        Ok(())
    }

    pub async fn store_points(&self, data: Points) -> Result<()> {
        let collection = self.database.collection::<Points>("points-data");
        let inserted = collection.insert_one(data, None).await?;
        collection
            .delete_many(doc! {"_id": { "$ne": inserted.inserted_id }}, None)
            .await?;

        Ok(())
    }

    pub async fn store_rain(&self, data: Rain) -> Result<()> {
        let collection = self.database.collection::<Rain>("rain-stats");
        let inserted = collection.insert_one(data, None).await?;
        collection
            .delete_many(doc! {"_id": { "$ne": inserted.inserted_id }}, None)
            .await?;

        Ok(())
    }
}

struct Mapper {
    pub client: Storage,
    pub logger: Logger,
}

impl Mapper {
    pub async fn new() -> Self {
        dotenv().ok();
        let client = Storage::new().await;
        let logger = Logger::new();
        Self { client, logger }
    }

    pub async fn run(&self) {
        let sleep = time::sleep(Duration::from_secs(WAIT_TIME));
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                () = &mut sleep => {
                    let now = Instant::now();

                     match self.compute().await {
                      Ok(_) => {
                        let elapsed_time = now.elapsed();
                        println!("Database updated, took {} seconds", elapsed_time.as_secs());
                      },
                      Err(e) => self.logger.log(&e).await,
                    }
                    sleep.as_mut().reset(Instant::now() + Duration::from_secs(WAIT_TIME));
                },
            }
        }
    }

    pub async fn compute(&self) -> Result<()> {
        let island: Vec<[f64; 2]> = vec![
            [-125.67796453456866, 48.82645842964928],
            [-124.74626480947109, 48.53388081242173],
            [-124.12134426214962, 48.352987949471014],
            [-123.75775412552608, 48.25473525182136],
            [-123.29190426297728, 48.28498699578307],
            [-123.14419576997425, 48.40581492060613],
            [-123.1328335782046, 48.66912781249633],
            [-123.16692015351316, 48.93853594737391],
            [-123.63277001606194, 49.31772215481024],
            [-124.20087960453596, 49.38433647100575],
            [-124.65536727531511, 49.70870880764804],
            [-125.09849275432498, 50.02363012281822],
            [-125.337098781484, 50.27844702316449],
            [-125.98474371234477, 50.45239434476167],
            [-126.62102645143548, 50.61128627109824],
            [-127.39365549176037, 50.83427177039019],
            [-127.82541877900061, 51.01332638323149],
            [-128.51851247693878, 50.9059763143982],
            [-128.65485877817255, 50.242127725727585],
            [-128.06402480615964, 49.7527745974472],
            [-127.15504946460135, 49.26584864698222],
            [-125.67796453456866, 48.82645842964928],
        ];
        //lets gen some geometry
        let island_geo = Polygon::new(LineString::from(island), vec![]);

        // now need to compute the grid of points

        let bound_rect = island_geo.bounding_rect().unwrap();

        let min_x = bound_rect.min().x;
        let min_y = bound_rect.min().y;
        let max_x = bound_rect.max().x;
        let max_y = bound_rect.max().y;
        //First need to get data

        //this is a vector of coords over a grid
        let grid = arange_grid(
            [min_y, min_x]..[max_y, max_x],
            [COORDINATE_STEP_Y, COORDINATE_STEP_X],
        );

        //need length of grid

        let x_width = ((max_x - min_x) / COORDINATE_STEP_X).ceil() as usize;
        let y_width = ((max_y - min_y) / COORDINATE_STEP_Y).ceil() as usize;

        let grid_points: Vec<Vec<f64>> = grid.into_iter().map(|x| vec![x[1], x[0]]).collect();

        //now we need to get the observations

        let current_observations = self.compute_points().await?;

        let coords = get_coordinates(&current_observations);
        let temps = get_temperatures(&current_observations)?;

        // ok so this is fucky... init to -infinity?? then max is a func that gets passed new and folded val
        let max_temp = temps.iter().cloned().fold(f64::NAN, f64::max);
        let min_temp = temps.iter().cloned().fold(f64::NAN, f64::min);

        // I want there to always be n steps between min and max, so create a linspace

        let steps = lin_space((min_temp)..(max_temp), NUMBER_OF_BANDS)
            .into_iter()
            .collect::<Vec<f64>>();

        let exponential_kernel = Exponential::default();

        let gp = GaussianProcess::builder(coords, temps)
            .set_noise(MODEL_NOISE)
            .set_kernel(exponential_kernel)
            .fit_prior()
            .train();

        // makes several prediction
        let outputs = gp.predict(&grid_points);

        let c = ContourBuilder::new(
            x_width.try_into().unwrap(),
            y_width.try_into().unwrap(),
            true,
        );

        let mut contour = c.contours(&outputs, &steps).unwrap();
        let g = colorgrad::turbo();

        let mut done: Option<MultiPolygon> = None;

        contour = contour
            .into_iter()
            .rev()
            .map(|mut feature| {
                let mut geo = feature.geometry().as_mut().unwrap();

                let mut multi: MultiPolygon = geo.value.clone().try_into().unwrap();

                multi = multi.map_coords(|Coord { x, y }| Coord {
                    x: remap_x(x, x_width as f64, min_x, max_x),
                    y: remap_y(y, y_width as f64, min_y, max_y),
                });

                match &done {
                    Some(multi_done) => {
                        multi = BooleanOps::difference(&multi, multi_done);
                        done = Some(BooleanOps::union(multi_done, &multi));
                    }
                    None => {
                        done = Some(multi.clone());
                    }
                }

                multi =
                    BooleanOps::intersection(&MultiPolygon::new(vec![island_geo.clone()]), &multi);

                geo.value = Value::from(&multi);
                let prop_temp = feature
                    .property("value")
                    .unwrap()
                    .as_f64()
                    .unwrap_or_default();

                feature.set_property(
                    "fill",
                    g.at((prop_temp - min_temp) / (max_temp - min_temp))
                        .to_hex_string(),
                );

                feature
            })
            .collect::<Vec<Contour>>();

        match self
            .client
            .store_intersection(Intersection {
                intersection: FeatureCollection {
                    bbox: Some(vec![min_x, min_y, max_x, max_y]),
                    // bbox: None,
                    features: contour,
                    foreign_members: None,
                },
            })
            .await
        {
            Ok(_) => (),
            Err(e) => self.logger.log(&e).await,
        }

        Ok(())
    }

    pub async fn compute_points(&self) -> Result<Vec<Station>> {
        let mut max_point: Option<Feature> = None;
        let mut min_point: Option<Feature> = None;

        let mut max_temp = -100.0;
        let mut min_temp = 100.0;

        let mut avg_temp = 0.0;
        let mut reporting_count = 0;

        let mut rain_total = 0.0;
        let mut rain_reporting_count = 0;
        let mut rain_currently_reporting_count = 0;

        let mut max_rain = 0.0;
        let mut max_rain_point: Option<Feature> = None;

        let body = reqwest::get("https://www.victoriaweather.ca/stations/latest/allcurrent.xml")
            .await?
            .text()
            .await?;

        //Now we can parse the data

        let json_builder = JsonConfig::new().explicit_array(false).finalize();
        let val = json_builder.build_from_xml(&body)?;

        let mut conditions: ConditionsWrapper = serde_json::from_value(val).unwrap();

        let current_observations =
            filter_observations(&mut conditions.current_conditions.current_observation);

        let points_data: Vec<Feature> = current_observations
            .iter()
            .map(|x| {
                let mut feat = Feature::from(Value::from(&Point::new(
                    x.longitude.parse::<f64>().unwrap() - 360.0,
                    x.latitude.parse::<f64>().unwrap(),
                )));
                feat.properties = Some(json!(x).as_object().unwrap().to_owned());

                let temp = x
                    .temperature
                    .as_ref()
                    .unwrap_or(&"".to_string())
                    .parse::<f64>()
                    .unwrap_or_default();

                feat.set_property("temperature", temp);

                avg_temp += temp;
                reporting_count += 1;
                if temp > max_temp {
                    max_temp = temp;
                    max_point = Some(feat.clone());
                }
                if temp < min_temp {
                    min_temp = temp;
                    min_point = Some(feat.clone());
                }

                if let Some(rain) = &x.rain {
                    let rain = rain.parse::<f64>().unwrap_or_default();

                    if rain > 0.0 {
                        rain_reporting_count += 1;
                    }

                    if rain > max_rain {
                        max_rain = rain;
                        max_rain_point = Some(feat.clone());
                    }
                    rain_total += rain;
                }
                if let Some(_rate) = &x.rain_rate {
                    let rate = _rate.parse::<f64>().unwrap_or_default();
                    if rate > 0.0 {
                        rain_currently_reporting_count += 1;
                    }
                }

                feat
            })
            .collect();

        let avg_rain = rain_total / rain_reporting_count as f64;

        match self
            .client
            .store_rain(Rain {
                max_rain: max_rain_point,
                average_rain: avg_rain,
                number_reporting: rain_currently_reporting_count,
            })
            .await
        {
            Ok(_) => (),
            Err(e) => self.logger.log(&e).await,
        }

        avg_temp /= reporting_count as f64;

        match self
            .client
            .store_points(Points {
                points: FeatureCollection {
                    features: points_data,
                    bbox: None,
                    foreign_members: None,
                },
                average_temp: avg_temp,
                min_point,
                max_point,
            })
            .await
        {
            Ok(_) => (),
            Err(e) => self.logger.log(&e).await,
        }
        Ok(current_observations)
    }
}

fn filter_observations(observations: &mut Vec<Station>) -> Vec<Station> {
    observations
        .drain(..)
        .filter(|observation| {
            observation.temperature.is_some()
                && (observation.latitude.parse::<f64>().unwrap_or_default() < MAX_LAT)
        })
        .collect()
}

fn get_coordinates(stations: &Vec<Station>) -> Vec<Vec<f64>> {
    stations
        .into_iter()
        .map(|x| {
            let lat = x.latitude.parse::<f64>().unwrap_or_default();
            let lon = x.longitude.parse::<f64>().unwrap_or_default() - 360.0;
            vec![lon, lat]
        })
        .collect::<Vec<Vec<f64>>>()
}

fn get_temperatures(stations: &Vec<Station>) -> Result<Vec<f64>> {
    let mut temperatures = Vec::new();
    for station in stations {
        match &station.temperature {
            Some(temp) => {
                let temp = temp.parse::<f64>()?;
                temperatures.push(temp);
            }
            None => continue,
        }
    }
    Ok(temperatures)
}

pub fn remap_x(curr: f64, max: f64, new_min: f64, new_max: f64) -> f64 {
    new_min + ((curr / max) * (new_max - new_min))
}
pub fn remap_y(curr: f64, max: f64, new_min: f64, new_max: f64) -> f64 {
    new_min + ((curr / max) * (new_max - new_min))
}

#[tokio::main]
async fn main() {
    let mapper = Mapper::new().await;

    mapper.run().await;
}
