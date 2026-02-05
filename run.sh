gdal_translate -b 1 -ot Float32 CDSM.tif CDSM_f32.tif
cargo run --release -- CDSM_f32.tif
python plot_arbres.py arbres_detectes.csv CDSM_f32.tif --hmin 10 --top 30 --save arbres_detectes.png


