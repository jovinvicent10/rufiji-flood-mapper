// Export Sentinel-1 images for the Rufiji flood mapper.
// Paste into the Google Earth Engine Code Editor (https://code.earthengine.google.com), edit the
// settings below, select Run, then open the Tasks tab and select Run on both exports.
// The GeoTIFFs appear in your Google Drive, in the folder named below.

// ---- Settings -------------------------------------------------------------
// Area: [west, south, east, north] in degrees. Keep it under about 40 x 40 km.
// Example: lower Rufiji floodplain.
var AREA = [38.95, -8.05, 39.12, -7.90];

var FLOOD_START = '2024-04-05', FLOOD_END = '2024-05-05';   // the flood you want to map
var REF_START   = '2023-08-01', REF_END   = '2023-10-31';   // a normal (dry-season) period

// Map projection in metres. For Tanzania (southern hemisphere) use the UTM zone for your longitude:
// 30-36 E -> 'EPSG:32736'   36-42 E -> 'EPSG:32737'   24-30 E -> 'EPSG:32735'
var CRS = 'EPSG:32737';
var FOLDER = 'flood_mapper';
// ---------------------------------------------------------------------------

var aoi = ee.Geometry.Rectangle(AREA);

function s1Median(start, end) {
  var col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);
  print('Images between ' + start + ' and ' + end + ':', col.size());
  return col.median().clip(aoi).toFloat();   // values are already in decibels, as the model expects
}

var flood = s1Median(FLOOD_START, FLOOD_END);
var ref = s1Median(REF_START, REF_END);

Map.centerObject(aoi, 11);
Map.addLayer(ref.select('VV'), {min: -25, max: 0}, 'Normal season (VV)', false);
Map.addLayer(flood.select('VV'), {min: -25, max: 0}, 'Flood period (VV) - dark = water');

Export.image.toDrive({
  image: flood, description: 'S1_flood', fileNamePrefix: 'S1_flood',
  folder: FOLDER, region: aoi, scale: 10, crs: CRS, maxPixels: 1e9
});
Export.image.toDrive({
  image: ref, description: 'S1_reference', fileNamePrefix: 'S1_reference',
  folder: FOLDER, region: aoi, scale: 10, crs: CRS, maxPixels: 1e9
});
