import flask
from flask import Flask, request, render_template
import sklearn
import numpy as np
import joblib

import locale
locale.setlocale( locale.LC_ALL, '' )

app = Flask(__name__, template_folder='tmplts')
mymodel = joblib.load('rm.pkl')

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def make_prediction():
    #print("make_prediction() starting")
    if request.method=='POST':
        entered_li = []
        print(request.form)

        # ---YOUR CODE FOR Part 2.2.3 ----

# ([('Month', '1'), ('Day_of_the_Week', '1'),
#  ('open', '1'), ('promo', 'on'), ('promo2', 'on'), ('state_holiday', 'on'), 
# ('SchoolHoliday', '0'), ('store', '1'), ('Assortment', '1'), ('StoreType', '1')])
        # Define "day", "open", "prn", "state", "school", and "store" variables
        day = int(request.form['Day_of_the_Week'])
        month = int(request.form['Month'])
        isopen = int(request.form['open'])
        schoolHoliday = int(request.form['SchoolHoliday'])
        store = int(request.form['store'])
        assortment = int(request.form['Assortment'])
        storeType = int(request.form['StoreType'])
        # promo = 1 if request.form['promo']=="on" else 0
        promo = 1 if 'promo' in request.form.keys() else 0
        # promo2 = 1 if request.form['promo2']=="on" else 0
        promo2 = 1 if 'promo2' in request.form.keys() else 0
        # stateHoliday = 1 if request.form['state_holiday']=="on" else 0
        stateHoliday = 1 if 'state_holiday' in request.form.keys() else 0
        entered_li = [day, isopen, promo,stateHoliday, schoolHoliday, store, storeType,assortment]

        print('state holiday', stateHoliday)
# DayOfWeek                               1.0
# Open                                    1.0
# Promo                                   0.0
# StateHoliday                            0.0
# SchoolHoliday                           0.0
# Store                                  79.0
# StoreType                               1.0
# Assortment                              1.0

    # CompetitionDistance                  3320.0
    # CompetitionOpenSinceMonth               0.0
    # CompetitionOpenSinceYear                0.0
# Promo2                                  0.0
    # Promo2SinceWeek                         0.0
    # Promo2SinceYear                         0.0
    # CompetitionOpenSinceMonth_missing       0.0
    # CompetitionOpenSinceYear_missing        0.0
    # CompetitionDistance_missing             0.0




        # print("DayOfWeek {}".format(request.form['Day_of_the_Week']))
        # for key_name in request.form.keys():
        #    print(key_name)

        # # --- THE END OF CODE FOR Part 2.2.3 ---

        # # StoreType, Assortment, ##CompetitionDistance... arbitrary values
        # for val in [1, 1, 290.0, 10.0, 2011.0, 1, 40.0, 2014.0, 0, 0, 0]:
        for val in [290.0, 10.0, 2011.0, promo2, 40.0, 2014.0, 0, 0, 0]:
            entered_li.append(val)
        # entered_li = entered_li + [290.0, 10.0, 2011.0, 1, 40.0, 2014.0, 0, 0, 0]
        # print(entered_li)
        # # ---YOUR CODE FOR Part 2.2.4 ----

        # Predict from the model
        # print(np.array(entered_li).reshape(-1,1))
        prediction = mymodel.predict(np.array(entered_li).reshape(1,-1))

        # # --- THE END OF CODE FOR Part 2.2.4 ---
        label = locale.currency( prediction, grouping=True )

        return render_template('index.html', label=label)


if __name__ == '__main__':

    # ---YOUR CODE FOR Part 2.2.1 ----

    #Load ML model



    # --- THE END OF CODE FOR Part 2.2.1 ---

    # start API
    app.run(host='0.0.0.0', port=8000, debug=True)
