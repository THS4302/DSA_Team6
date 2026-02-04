def generate_gray_code(n):
    if n <= 0:
        return []

    gray_codes = ['0', '1']

    for i in range(2, n + 1):
        # Mirror the current list
        mirrored = gray_codes[::-1]

        # Add '0' to the original list
        gray_codes = ['0' + code for code in gray_codes]

        # Add '1' to the mirrored list
        mirrored = ['1' + code for code in mirrored]

        # Combine
        gray_codes += mirrored

    return gray_codes


# example usage
if __name__ == "__main__":
    bits = 3
    codes = generate_gray_code(bits)
    print(f"{bits}-bit Gray Code Sequence:")
    for c in codes:
        print(c)
