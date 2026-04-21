const getCSSVar = (varName) => {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
};

const traceBuilders = {

    scatter: (data) => [{
        x: data.x,
        y: data.y,
        type: 'scatter',
        mode: 'markers',
        opacity: 0.5,
        marker: { color: 'white', size: 4, line: { color: 'rgb(0,0,0)', width: 1 } }
    }],

    // { epoch: [], train: [], val: [] | None }
    learning_curves: (data) => {
        const traces = [{
            x: data.epoch,
            y: data.train,
            type: 'scatter',
            mode: 'lines',
            name: 'Train loss',
            line: { width: 2 }
        }];
        if (data.val) traces.push({
            x: data.epoch,
            y: data.val,
            type: 'scatter',
            mode: 'lines',
            name: 'Val loss',
            line: { width: 2, dash: 'dash' }
        });
        return traces;
    },

    // { epoch: [], ...<parameter_name>: [] }
    parameters: (data) => {
        const traces = [];

        for (let parameter_name of Object.keys(data)) {
            if (parameter_name == "epoch") {
                continue;
            }
            traces.push({
                x: data.epoch,
                y: data[parameter_name],
                type: 'scatter',
                mode: 'lines',
                name: parameter_name,
                line: { width: 2 }
            })

        }

        return traces;
    },

    // { x: [], y: [], z: [[]] }
    loss_landscape: (data) => [{
        x: data.x,
        y: data.y,
        z: data.z,
        type: 'surface',
        colorscale: 'Viridis',
        contours: {
            z: { show: true, usecolormap: true, highlightcolor: '#42f462', project: { z: true } }
        }
    }],
};

const layoutBuilders = {
    base: (title) => ({
        title: title || "Plot",
        autosize: true,
        paper_bgcolor: getCSSVar("--background-color"),
        plot_bgcolor:  getCSSVar("--background-color"),
        font: { color: getCSSVar("--text-color") },
        xaxis: { gridcolor: getCSSVar("--text-color") },
        yaxis: { gridcolor: getCSSVar("--text-color") },
        showlegend: true
    }),

    learning_curves: (title) => {
        const layout = layoutBuilders.base(title || "Learning Curves")
        layout["xaxis"]["title"] = "Epoch"
        layout["yaxis"]["title"] = "Loss"

        return layout;
    },

    parameters: (title) => {
        const layout = layoutBuilders.base(title || "Parameters")
        layout["xaxis"]["title"] = "Epoch"

        return layout;
    },

}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.plotly-container[data-plotly]').forEach(el => {
        const data   = JSON.parse(decodeURIComponent(el.dataset.plotly));
        const title  = el.dataset.title;
        const type   = el.dataset.type;

        const traceBuilder = traceBuilders[type];
        const layoutBuilder = type in layoutBuilders ? layoutBuilders[type] : layoutBuilders.base;
        if (!traceBuilder) {
            console.warn(`[plots.js] Unknown plot type: "${type}"`);
            return;
        }

        Plotly.newPlot(el.id, traceBuilder(data), layoutBuilder(title), { responsive: true });
    });
});