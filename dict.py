
# Function to make results friendly
def process_file(file):

    char_list = [ch for ch in open(file).read() if ch != '\n' if ch != ' ']

    for i in range(0, len(char_list)):
        if char_list[i] == '{':
            j = 0
            for j in range(i, len(char_list)):
                if char_list[j] == '}':
                    item = ''.join(char_list[i:j+1])

                    with open("out.json", 'a') as test:
                        test.write("{}\n".format(item))

                    break
            i = j

process_file("log.txt")
