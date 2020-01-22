from psrdada import Reader
import numpy as np

def read_buffer(reader):
        """
        Reads a psrdada buffer as unsigned shorts and returns the visibilities.
        """
        page = reader.getNextPage()
        data = np.asarray(page)
        reader.markCleared()
        data.dtype = np.uint16
        dy = data.reshape(200000,2048).T 
        # We want dy to have dimensions (baselines, time, frequency, polarization)
        return dy

"""
Example Use:
------------
rkey=u'0xrada'
wkey=u'0xwada'
# With Reader(rkey) as reader, Writer(wkey) as writer:#defines a psrdada ring buffer to read
# Have to add the context manager info to the classes to use
reader = Reader(rkey)
writer = writer(wkey)
header = reader.getHeader()
# This returns a dictionary of keywords in the first header the reader finds
# after the call.
# What will these be and which ones are important?

while reader.isConnected and not reader.isEndOfData: #as long as the reader is connected...
    data = read_buffer(reader)
    # Do stuff here to the data

    # Write out the data
    # send 10 datasets, separated by an EOD

    for ndataset in range(10):
        npages = randint(1, 10)

        # setting a new header also resets the buffer: isEndOfData = False
        writer.setHeader({
            'DATASET': str(ndataset),
            'PAGES': str(npages),
            'MAGIC': random_string(20)
        })
        print(writer.header)

        for npage in range(npages):
            page = writer.getNextPage()
            data = np.asarray(page)
            data.fill(npage)

            # marking a page filled will send it on the ringbuffer,
            # and then we cant set the 'EndOfData' flag anymore.
            # so we need to treat the last page differently
            if npage < npages - 1:
                writer.markFilled()
            else:
                time.sleep(0.5)

        # mark the last page with EOD flag
        # this will also mark it filled, and send it
        writer.markEndOfData()


# Send a message to the reader that we're done
writer.setHeader({'QUIT': 'YES'})

reader.disconnect() #disconnect from the buffer
writer.disconnect()
"""