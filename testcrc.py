import crcmod

crc8_maxim = crcmod.predefined.Crc('crc-8-maxim')

def calculate_crc(byte_list):
    byte_data = bytes(byte_list)  
    crc8_maxim.update(byte_data)
    return crc8_maxim.crcValue

byte_list = [0x01, 0x02, 0x03, 0x04]  
crc = calculate_crc(byte_list)
print("CRC-8/MAXIM checksum:", hex(crc))
