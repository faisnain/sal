from flask import Flask, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask_cors import CORS
import pandas.api.types
import os
import shutil
import joblib
import warnings
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet, GammaRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn import preprocessing
from lazypredict.Supervised import LazyRegressor
import lazypredict
import datetime

warnings.warn('ignore', category=FutureWarning)
np.random.seed(1)

app = Flask(__name__)
cors = CORS(app)
ALLOWED_EXTENSIONS = (['csv'])
app.secret_key = "abcdef"
lazypredict.Supervised.REGRESSORS = lazypredict.Supervised.REGRESSORS[:10]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/project_list', methods=["GET", "POST"])
def project_list():
    if request.method == 'GET':
        UPLOAD_FOLDER_1 = 'projects/'
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1

        path = os.listdir('projects/')
        lis = []
        for i in range(len(path)):
            lis.append(path[i])
        available_projects = {"Projects": lis}
        return available_projects


@app.route('/load_project', methods=["GET", "POST"])
def select_project():
    if request.method == 'POST':
        project_name = request.form['project_name']
        return jsonify("Project Loaded Successfully!")


@app.route('/create_project', methods=["GET", "POST"])
def create_project():
    if request.method == 'POST':
        project_name = request.form['project_name']
        UPLOAD_FOLDER_1 = 'projects/' + project_name + '/'
        if os.path.isdir(UPLOAD_FOLDER_1):
            return jsonify("Project " + project_name + " already exists")
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        return jsonify("Project " + project_name + " Created Successfully")


@app.route('/load_data', methods=["GET", "POST"])
def data_load():
    if request.method == 'POST':
        project_name = request.form['project_name']
        UPLOAD_FOLDER_1 = 'projects/' + project_name + '/' + 'data/'
        UPLOAD_FOLDER_2 = 'projects/' + project_name + '/' + 'models/'
        if os.path.isdir(UPLOAD_FOLDER_1):
            shutil.rmtree('projects/' + project_name + '/' + 'data')
        if os.path.isdir(UPLOAD_FOLDER_2):
            shutil.rmtree('projects/' + project_name + '/' + 'models')
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        if not os.path.isdir(UPLOAD_FOLDER_2):
            os.mkdir(UPLOAD_FOLDER_2)
        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        if 'file' not in request.files:
            return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER_1'], file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        df = pd.read_csv(filepath)
        cols = list(df.columns)
        cols_response = {'columns': cols}
        return cols_response


@app.route('/train', methods=["GET", "POST"])
def model_training():
    if request.method == 'POST':
        project_name = request.form['project_name']
        file = os.listdir('projects/' + project_name + '/' + 'data/')
        dataset = pd.read_csv('projects/' + project_name + '/' + 'data/' + file[0])
        target = request.form['target']

        if pandas.api.types.is_numeric_dtype(dataset[target]):
            dataset['date'] = pd.to_datetime(dataset['date'])
            dataset['month'] = [i.month for i in dataset['date']]
            dataset['year'] = [i.year for i in dataset['date']]
            dataset['day_of_week'] = [i.dayofweek for i in dataset['date']]
            dataset['day_of_year'] = [i.dayofyear for i in dataset['date']]
            dataset['day'] = [i.day for i in dataset['date']]
            le = preprocessing.LabelEncoder()
            dataset['category'] = le.fit_transform(dataset['category'])
            joblib.dump(le, 'projects/' + project_name + '/' + 'product_encoder.joblib', compress=9)
            # load it when test
            X = dataset.drop([target, 'date'], axis=1)
            y = dataset[target]
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
            d = {'BaggingRegressor': BaggingRegressor(), 'ElasticNetCV': ElasticNetCV(), 'ElasticNet': ElasticNet(),
                 'ExtraTreeRegressor': ExtraTreeRegressor(), 'GammaRegressor': GammaRegressor(),
                 'ExtraTreesRegressor': ExtraTreesRegressor(), 'DecisionTreeRegressor': DecisionTreeRegressor(),
                 'DummyRegressor': DummyRegressor(), 'AdaBoostRegressor': AdaBoostRegressor(),
                 'BayesianRidge': BayesianRidge()}
            models, _ = reg.fit(train_x, test_x, train_y, test_y)
            model_names = []
            mae_list = []
            r2_list = []
            adj_r2_list = []
            n = len(test_x)
            p = len(test_x.columns)
            for i in range(0, 3):
                model = d[models.index[i]]
                model.fit(train_x, train_y)
                predictions = model.predict(test_x)
                r2_value = r2_score(test_y, predictions)
                mae_value = mean_absolute_error(test_y, predictions)
                adj_r2 = 1 - (1 - r2_value) * (n - 1) / (n - p - 1)
                model_names.append(models.index[i])
                mae_list.append(mae_value)
                r2_list.append(r2_value)
                adj_r2_list.append(adj_r2)
                joblib.dump(model, 'projects/' + project_name + '/' + 'models/' + models.index[i] + ".pkl")
            modelDetails = {'Models': model_names, 'MAE': mae_list,
                            'r2 score': r2_list, 'Adjusted r2 score': adj_r2_list}
        else:
            return jsonify('Target Feature name should be Numeric')
        return modelDetails


@app.route('/model_list', methods=["GET", "POST"])
def model_list():
    project_name = request.form['project_name']
    path = os.listdir('projects/' + project_name + '/' + 'models/')
    lis = []
    for i in range(len(path)):
        lis.append(path[i][:-4])
    available_models = {"Models": lis}
    return available_models


@app.route('/test', methods=['GET', 'POST'])
def model_test():
    if request.method == 'POST':
        project_name = request.form['project_name']
        UPLOAD_FOLDER = 'projects/' + project_name + '/' + 'test_data/'
        file = request.files['test_file']
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree('projects/' + project_name + '/' + 'test_data')
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)
        if 'test_file' not in request.files:
            return jsonify('No file part')
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        model_name = request.form['model_name']
        data_test = pd.read_csv(filepath)
        data = data_test.copy()
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = [i.month for i in data['date']]
        data['year'] = [i.year for i in data['date']]
        data['day_of_week'] = [i.dayofweek for i in data['date']]
        data['day_of_year'] = [i.dayofyear for i in data['date']]
        data['day'] = [i.day for i in data['date']]
        le = joblib.load('projects/' + project_name + '/' + 'product_encoder.joblib')
        data['category'] = le.transform(data['category'])
        X = data.drop(["date"], axis=1)
        path = os.listdir('projects/' + project_name + '/' + 'models/')
        lis = []
        for i in range(len(path)):
            lis.append(path[i][:-4])
        available_models = {"Models": lis}
        if model_name in available_models['Models']:
            model = joblib.load('projects/' + project_name + '/' + 'models/' + model_name + '.pkl')
        else:
            return jsonify("Model not found!")
        data_test['sales'] = model.predict(X)
        data_test['sales'] = data_test['sales'].apply(np.ceil)
        data_test['sales'] = data_test['sales'].astype(int)
        if os.path.isfile('projects/' + project_name + '/' + "Output.csv"):
            os.remove('projects/' + project_name + '/' + "Output.csv")
        data_test.to_csv('projects/' + project_name + '/' + "Output.csv", index=False)
        return data_test.to_dict()
    else:
        return jsonify('GET method is not supported')


@app.route('/download', methods=['GET', 'POST'])
def download():
    if request.method == 'GET':
        project_name = request.form['project_name']
        if os.path.isfile('projects/' + project_name + '/' + "Output.csv"):
            return send_from_directory(os.getcwd(), path='projects/' + project_name + '/' + "Output.csv",
                                       as_attachment=True)
        else:
            return jsonify("No Predictions")


@app.route('/plot_chart', methods=['GET', 'POST'])
def plot_chart():
    project_name = request.form['project_name']
    if os.path.isfile('projects/' + project_name + '/' + "Output.csv"):
        df = pd.read_csv('projects/' + project_name + '/' + "Output.csv")
        df['date'] = pd.to_datetime(df['date'])
        tdf1 = df
        tdf_ = tdf1.groupby('item')['sales'].sum()
        tdf_.sort_values(ascending=False, inplace=True)
        ind = list(tdf_.index)[:8]
        d = {}
        for i in ind:
            tdf = tdf1[tdf1['item'] == i]
            tdf.set_index("date", inplace=True)
            tdf = tdf.resample("W")['sales'].sum()
            tdf = tdf.reset_index()
            tdf['date'] = tdf['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            d["item " + str(i)] = {"date": list(tdf['date']), "sales": list(tdf['sales'])}
        return d
    else:
        return jsonify("No Predictions to plot")


@app.route('/kpi', methods=['GET', 'POST'])
def kpi_data():
    project_name = request.form['project_name']
    if os.path.isfile('projects/' + project_name + '/' + "Output.csv"):
        df = pd.read_csv('projects/' + project_name + '/' + "Output.csv")
        print(df)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")
        grouped = df.groupby('item').resample('W')['sales'].sum()
        grouped = grouped.reset_index()
        total_sale = grouped.groupby('date').agg({'sales': "sum"}).reset_index()
        total_sale['date'] = total_sale['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        total_weekly_sales = dict(zip(total_sale.date, total_sale.sales))
        df_max = grouped[grouped.groupby('date').sales.transform('max') == grouped.sales]
        df_max = df_max.sort_values(by='date')
        df_max['date'] = df_max['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        top_selling = df_max.groupby('date').apply(
            lambda df_max: df_max[['sales', 'item']].to_dict(orient='list')).to_dict()
        df_item = grouped.groupby("item").sum()
        df_item = df_item.reset_index()
        df_item['labels'] = df_item.item.apply(lambda x: "item " + str(x))
        item_total_sales = dict(zip(df_item.labels, df_item.sales))
        return jsonify(total_sales_weekly=total_weekly_sales, top_selling=top_selling,
                       item_total_sales=item_total_sales)


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate_performance():
    project_name = request.form['project_name']
    path = os.listdir('projects/' + project_name + '/' + 'data/')
    df = pd.read_csv('projects/' + project_name + '/' + 'data/' + path[0])
    df['date'] = pd.to_datetime(df['date'])
    end_date = df['date'].max()
    range_date = datetime.timedelta(days=60)
    split_date = end_date - range_date
    data_test = df[df['date'] >= split_date]
    data_test.reset_index(drop=True, inplace=True)
    data = data_test.copy()
    # data.reset_index(drop=True, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = [i.month for i in data['date']]
    data['year'] = [i.year for i in data['date']]
    data['day_of_week'] = [i.dayofweek for i in data['date']]
    data['day_of_year'] = [i.dayofyear for i in data['date']]
    data['day'] = [i.day for i in data['date']]
    le = joblib.load('projects/' + project_name + '/' + 'product_encoder.joblib')
    data['category'] = le.transform(data['category'])
    X = data.drop(["date", "sales"], axis=1)
    path = os.listdir('projects/' + project_name + '/' + 'models/')
    d = {}
    for i in path:
        model = joblib.load('projects/' + project_name + '/' + 'models/' + i)
        data_test[i[:-4]] = model.predict(X)
        data_test[i[:-4]] = data_test[i[:-4]].apply(np.ceil)
        data_test[i[:-4]] = data_test[i[:-4]].astype(int)
    ind = data['item'].unique()[:5]
    for i in ind:
        tdf = data_test[data_test['item'] == i]
        tdf['date'] = tdf['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        d["item " + str(i)] = {"date": list(tdf['date']), "actual_sales": list(tdf['sales']),
                               path[0][:-4]: list(tdf[path[0][:-4]]), path[1][:-4]: list(tdf[path[1][:-4]]), path[2][:-4]: list(tdf[path[2][:-4]])}
    data_test.to_csv('projects/' + project_name + "/evaluation.csv", index=False)
    return jsonify(d)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
