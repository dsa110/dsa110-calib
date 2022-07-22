systemctl --user stop realtime_calibration.service
systemctl --user stop dask_worker_highmem.service
systemctl --user stop dask_worker_lowmem.service
systemctl --user stop dask_scheduler.service

sleep 10

systemctl --user start dask_scheduler.service
systemctl --user start dask_worker_highmem.service
systemctl --user start dask_worker_lowmem.service
systemctl --user start realtime_calibration.service
