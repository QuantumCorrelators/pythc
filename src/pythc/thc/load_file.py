from pythc.thc.thc_base import THC, Mode, THC_ERI, THC_ERI_Unrestricted

class FileTHC(THC):
    def __init__(self, file: str):
        self.file = file

    def build(self, mode: Mode) -> THC_ERI:
        return THC_ERI.from_file(self.file)

    def build_unrestricted(self, mode: Mode) -> THC_ERI_Unrestricted:
        return THC_ERI_Unrestricted.from_file(self.file)

