from subprocess import check_call

check_call(["dot", "-Tpng", "Results.dot", "-o", "Results.png"])
print("Conversion complete")