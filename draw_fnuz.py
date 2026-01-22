import numpy


class FloatPoint8:
    def __init__(self):
        self.exp = None
        self.mantissa = None
        self.sign = None

    def decode(self, value):
        pass

class Float8E4M3FN:
    def __init__(self):
        self.exp = None
        self.mantissa = None
        self.sign = None

    def decode(self, value):
        assert value<=255 and value>=0, "Value out of range for Float8E4M3FN"

        self.sign = (value >> 7) & 0x1
        self.exp = (value >> 3) & 0xF
        self.mantissa = value & 0x7

        # NaN
        if self.exp == 0xF and self.mantissa == 0x7:
            return numpy.nan
        elif self.exp == 0 and self.mantissa == 0:
            return 0
        elif self.exp == 0: # Denormalized number
            sign = (-1) ** self.sign 
            exponent = (2**(-6)) * 1.0 
            mantissa = 0
            for i in range(3):
                mantiassa_exp = 2**(i-3)
                bit_i = self.mantissa >> i & 0x1
                mantissa += (bit_i * 1.0) * mantiassa_exp
            return sign * exponent * mantissa
        else: # Normalized number
            sign = (-1) ** self.sign
            exponent = 0
            for i in range(3, 7):
                exp_i = (2**(i-3)) * 1.0
                # print("exp_i:", exp_i)
                # 3 -> 0 4->1
                bit_i = self.exp >> (i-3) & 0x1
                exponent += bit_i * exp_i
                # print("bit_i:", bit_i, "exp_i:", exp_i)
            exponent -= 7 # Bias
            mantissa = 1
            for i in range(3):
                mantissa_exp = 2**(i-3)
                bit_i = self.mantissa >> i & 0x1
                mantissa += bit_i * mantissa_exp
            return sign * (pow(2.0, exponent)) * mantissa
        
class Float8E4M3FNUZ:
    def __init__(self):
        self.exp = None
        self.mantissa = None
        self.sign = None

    def decode(self, value):
        assert value<=255 and value>=0, "Value out of range for Float8E4M3FNUZ"

        self.sign = (value >> 7) & 0x1
        self.exp = (value >> 3) & 0xF
        self.mantissa = value & 0x7

        # NaN
        if self.sign == 1 and self.exp == 0 and self.mantissa == 0:
            return numpy.nan
        elif self.exp == 0 and self.mantissa == 0:
            return 0
        elif self.exp == 0: # Denormalized number
            sign = (-1) ** self.sign 
            exponent = (2**(-7)) * 1.0 
            mantissa = 0
            for i in range(3):
                mantiassa_exp = 2**(i-3)
                bit_i = self.mantissa >> i & 0x1
                mantissa += (bit_i * 1.0) * mantiassa_exp
            return sign * exponent * mantissa
        else: # Normalized number
            sign = (-1) ** self.sign
            exponent = 0
            for i in range(3, 7):
                exp_i = (2**(i-3)) * 1.0
                # print("exp_i:", exp_i)
                # 3 -> 0 4->1
                bit_i = self.exp >> (i-3) & 0x1
                exponent += bit_i * exp_i
                # print("bit_i:", bit_i, "exp_i:", exp_i)
            exponent -= 8 # Bias
            mantissa = 1
            for i in range(3):
                mantissa_exp = 2**(i-3)
                bit_i = self.mantissa >> i & 0x1
                mantissa += bit_i * mantissa_exp
            return sign * (pow(2.0, exponent)) * mantissa

for i in range(256):
    f8 = Float8E4M3FN()
    f8_fnuz = Float8E4M3FNUZ()
    value = f8.decode(i)
    value_fnuz = f8_fnuz.decode(i)
    print(f"Float8E4M3FN({i}) = {value}, Float8E4M3FNUZ({i}) = {value_fnuz}")