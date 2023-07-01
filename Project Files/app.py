from flask import Flask, render_template, send_from_directory, request
import plotly.graph_objs as go
from prophet.plot import plot_components_plotly
import json
import pickle
from datetime import datetime, timedelta
import plotly
import pandas as pd

app = Flask(__name__)

model_path = 'models/fbprophet.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
 
    # Generate future dates
    future = model.make_future_dataframe(periods=365)
    future.tail()

    # Make predictions
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    # Get prediction for the next day
    next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    prediction = forecast[forecast['ds'] == next_day]['yhat'].item()

    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.update_layout(title_text="Time Series Plot of Bitcoin Open Price")

    # Configure the date range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # Convert the figure to JSON
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    fig_multi = plot_components_plotly(model, forecast)
    plot_json_multi = json.dumps(fig_multi, cls=plotly.utils.PlotlyJSONEncoder)

    if request.method == 'POST':
        selected_date = request.form['selected_date']
        if len(selected_date) > 0:
            selected_date = datetime.fromisoformat(selected_date).strftime('%Y-%m-%d')
            selected_date_data = {'ds': [selected_date]}
        
            # Create a DataFrame for the selected date
            selected_date_df = pd.DataFrame(selected_date_data)

            # Make prediction for the selected date
            prediction = model.predict(selected_date_df)['yhat'].values[0]


            ref_date = datetime.today().strftime('%Y-%m-%d')
            ref_date_data = {'ds': [ref_date]}
        
            # Create a DataFrame for the reference date i.e. today
            ref_date_df = pd.DataFrame(ref_date_data)
            ref = model.predict(ref_date_df)['yhat'].values[0]

            fig_specific = go.Figure(go.Indicator(
            mode = "number+delta",
            value = prediction,
            number = {'prefix': "$"},
            delta = {'position': "top", 'reference': ref},
            domain = {'x': [0, 1], 'y': [0, 1]}))

            fig_specific.update_layout(paper_bgcolor = "lightgray")
            plot_json_specific = json.dumps(fig_specific, cls=plotly.utils.PlotlyJSONEncoder)

            prediction = round(float(prediction), 2)
            
            return render_template('index.html', 
                                   plot_json=plot_json, 
                                   prediction=prediction, 
                                   plot_json_multi=plot_json_multi, 
                                   plot_json_specific = plot_json_specific, 
                                   selected_date=selected_date
                                   )

    # Render the HTML template with the plot
    return render_template('index.html', 
                           plot_json=plot_json, 
                           plot_json_multi=plot_json_multi,
                           selected_date=None,
                           plot_json_specific=None,
                           prediction=None
                           )

@app.route('/models/<path:filename>')
def download_model(filename):
    return send_from_directory('models', filename)


if __name__ == '__main__':
    app.run(debug=True)
