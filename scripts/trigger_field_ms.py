from dsautils import dsa_store
from astropy.time import Time


def trigger_field_ms():
    etcd = dsa_store.DsaStore()
    now = Time.now()
    nowmjd = now.mjd
    nowstr = now.strftime("%H:%M:%S")
    
    etcd.put_dict(
        "/cmd/cal",
        {
            "cmd": "field",
            "val": {
                "trigname": f"field{nowstr}",
                "mjds": nowmjd
    }})

    
if __name__ == "__main__":
    trigger_field_ms()


