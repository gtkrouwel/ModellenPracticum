import T_soil
import pandas as pd


def get_prop_time_data(circuit):
    file = open("Propagation (" + str(circuit) + ").csv", 'r')
    data = []
    first_line = True
    for line in file:
        if first_line:
            first_line = False
            continue
        data.append((pd.to_datetime(line[0:19]), float(line[20:-1])))
    file.close()
    return data


def get_temp_data(circuit, start, end):
    temps = T_soil.T_soil(
        circuit,
        pd.Timestamp(int(start[0:4]), int(start[5:7]), int(start[8:10])),
        pd.Timestamp(int(end[0:4]), int(end[5:7]), int(end[8:10]))
    )
    file = open("Climate_Data_Store_" + start + "_" + end + "_" + str(circuit) + ".csv", 'r')
    dates = []
    first_line = True
    for line in file:
        if first_line:
            first_line = False
            continue
        dates.append(pd.to_datetime(line[0:19]))
    data = list(zip(dates, temps))
    file.close()
    return data


def get_load_data(circuit):
    file = open("Power (" + str(circuit) + ").csv", 'r')
    data = []
    first_line = True
    for line in file:
        if first_line:
            first_line = False
            continue
        s = 0
        string = ""
        for c in line:
            if c == ';':
                s += 1
                if s == 3:
                    break
            elif c == '\n':
                break
            string += c
        data.append((pd.to_datetime(string[0:16].replace('/', '-').replace(';', ' ') + ":00"), float(string[17:].replace(',', '.'))))
    file.close()
    return data


def write_data(circuit, data_tuple):
    file = open("Parsed data (" + str(circuit) + ").csv", 'w')
    n = len(data_tuple)
    indices = [0] * n
    while all([el[0] < len(el[1]) for el in zip(indices, data_tuple)]):
        dates = [data_tuple[i][indices[i]][0] for i in range(0, n)]
        if all([dates[0] == date for date in dates]):
            line = dates[0].strftime('%Y-%m-%d %X')
            for i in range(0, n):
                line += '\t' + str(data_tuple[i][indices[i]][1])
            file.write(line + '\n')
            indices = [i + 1 for i in indices]
        else:
            indices[dates.index(min(dates))] += 1
    file.close()


circuit_number = 1358


prop_time_data = get_prop_time_data(circuit_number)
start_date = prop_time_data[0][0].strftime("%Y-%m-%d")
end_date = prop_time_data[-1][0].strftime("%Y-%m-%d")

temp_data = get_temp_data(circuit_number, start_date, end_date)

load_data = get_load_data(circuit_number)

write_data(circuit_number, (temp_data, prop_time_data, load_data))
