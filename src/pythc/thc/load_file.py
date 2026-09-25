from pythc.thc.thc_base import THC, Mode, ThcEri, ThcEriUnrestricted

class FileTHC(THC):
    def __init__(self, file: str):
        self.file = file

    def build(self, mode: Mode) -> ThcEri:
        return ThcEri.from_file(self.file)

    def build_unrestricted(self, mode: Mode) -> ThcEriUnrestricted:
        return ThcEriUnrestricted.from_file(self.file)

