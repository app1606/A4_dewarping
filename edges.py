from scipy import ndimage
import numpy as np
import cv2
from tqdm import tqdm


lambda_ = 0.95
d_min = 1
edge_dist = 1000

def edges(path, points, col_number=100, eta=5, N_max=5, rotation=False):
    
    def Var(data): #variance calculation
        return np.mean( (data - data.mean()) ** 2)

    
    def find_best_split(data):# best cut choice (cut, that maximizes variance)
        def split(i):
            if i == 0:
                return -1
            else:    
                return i * data[:i].mean() ** 2 + (len(data) - i) * (data[i:].mean() ** 2) - len(data) * data.mean() ** 2

        res = np.vectorize(split)(np.arange(data.shape[0]))
        max_ind = np.argmax(res)
        d_max = res[max_ind]

        return max_ind, d_max

    def best_cut(data, le, re, split_ind): #returns new cut candidates after one cut is performed 
        segments = [data[le:split_ind], data[split_ind:re]]

        cands = []

        left_ind, left_d = find_best_split(segments[0])
        right_ind, right_d = find_best_split(segments[1])

        cands.append((le + left_ind, left_d))
        cands.append((split_ind + right_ind, right_d))

        return cands


    def it_cut(data): #iterative cut algo

        Points = np.array([])
        cand_cut = []
        sigma = Var(data)

        max_ind, s_int = find_best_split(data)

        Points = np.append(Points, max_ind)

        le = 0
        re = len(data)


        cand_cut = best_cut(data, le, re, max_ind)

        x, d = max(cand_cut, key = lambda p: p[1])

        Points = np.append(Points, x)
        s_int += d
        cand_cut.remove((x, d))

        while len(Points) <= N_max and d >= d_min and s_int / sigma > lambda_:
            left = Points[np.where(Points < x)]
            right = Points[np.where(Points > x)]

            if left.size == 0:
                le = 0
            else:
                le = np.max(left)

            if right.size == 0:
                re = len(data)
            else:
                re = np.min(right) 
            cand_cut += best_cut(data, int(le), int(re), x)

            x, d = max(cand_cut, key = lambda p: p[1])
            Points = np.append(Points, x)
            s_int += d
            cand_cut.remove((x, d))


        return Points
    
    
    def line_detection(N, image, points , eta): #longest top and bottom lines choice
        top_left, top_right, bottom_left, bottom_right = points
        x_left = min(top_left[0], bottom_left[0])
        x_right = max(top_right[0], bottom_right[0])

        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        
        xs = np.random.choice(x_right - x_left, N) + x_left
        xs = np.sort(xs)
        
        
        print(x_right, img.shape)
        brightness = img[:, :, 2]
        saturation = img[:, :, 1]

        br_map = cv2.Sobel(brightness, ddepth=cv2.CV_32F, dx=0, dy=1).T
        sat_map = cv2.Sobel(saturation, ddepth=cv2.CV_32F, dx=0, dy=1).T

        sum_map = br_map + sat_map

        cut_points = []

        cut_dict = {}

        
        for col in tqdm(xs):
            sat_res = it_cut(img[:, col][:, 1])
            br_res = it_cut(img[:, col][:, 2])

            cut_points += list(zip([col] * len(sat_res), sat_res))
            cut_points += list(zip([col] * len(br_res), br_res))
            cut_dict[col] = list(np.concatenate((sat_res, br_res)))

        
        connection = {} #dict of point's best neighbor

        for i, x in enumerate(xs):
            if i == N - 1:
                break
            for j, y in enumerate(cut_dict[x]):
                g = -1e6
                f = False
                x_con, y_con = -1, -1
                for num in range(i + 1, N - 1):
                    for elem in cut_dict[xs[num]]:
                        if abs(elem - y) < eta and sum_map[xs[num], int(elem)] > g:

                            g = sum_map[xs[num], int(elem)]
                            x_con = xs[num]
                            y_con = int(elem)
                            f = True
                    if f:
                        break
                connection[(x, int(y))] = (x_con, y_con)

        H = []
        used = {}

        for i in connection:
            used[i] = 0

        for i in connection: #lines creation
            if connection[i] == (-1, -1):
                used[i] = 1
                continue

            if used[i]:
                for h in H:
                    if h[-1] == i:
                        h.append(connection[i])
                        used[connection[i]] = 1

            if not used[i]:
                H.append([i, connection[i]])
                used[i] = 1
                used[connection[i]] = 1

        return H

    def choose_best_line(lines, left_point, edge_dist):
        leng = 0 
        ans = []

        for line in lines: 
            flag = False

            for x, y in line: #check whether line is close to top/bottom left points
                if (x - left_point[0]) ** 2 + (y - left_point[1]) ** 2 < edge_dist:
                    flag = True

            if not flag:
                continue


            if len(line) > top_leng: 
                ans = line
                leng = len(line)
        return ans
        
    top_left, top_right, bottom_left, bottom_right = points
    
    full_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if rotation:
        full_image = cv2.rotate(full_image, cv2.cv2.ROTATE_90_CLOCKWISE)
        
    h = full_image.shape[0] // 2
    
    top_lines = line_detection(col_number, full_image[:h, :, :], points, eta)
    bot_lines = line_detection(col_number, full_image[h:, :, :], points, eta)
    
    
    top_leng = 0
    bot_leng = 0 
    top_ans = choose_best_line(top_lines, top_left, edge_dist)
    bot_ans = choose_best_line(bot_lines, np.array([bottom_left[0], bottom_left[1] - h]), edge_dist)
    bot_ans = [ (x, y + h - 1) for (x, y) in bot_ans ] #fix

    top_ans = [elem for elem in top_ans if elem[0] >= top_left[0] and elem[0] <= top_right[0]] #cut extra points
    bot_ans = [elem for elem in bot_ans if elem[0] >= bottom_left[0] and elem[0] <= bottom_right[0]]
    
    top_ans += [tuple(top_left), tuple(top_right)] 
    bot_ans += [tuple(bottom_left), tuple(bottom_right)]
    return top_ans, bot_ans
    