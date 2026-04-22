import Plotly from "plotly.js-dist-min"

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

    learning_curves: (data, smooth = false) => {
        return traceBuilders.parameters(data, smooth);
    },

    parameters: (data, smooth = false) => {
        const colorway = [
            'rgba(31, 119, 180, 1)',
            'rgba(255, 127, 14, 1)',
            'rgba(44, 160, 44, 1)',
            'rgba(214, 39, 40, 1)',
            'rgba(148, 103, 189, 1)',
            'rgba(140, 86, 75, 1)',
            'rgba(227, 119, 194, 1)',
            'rgba(127, 127, 127, 1)',
            'rgba(188, 189, 34, 1)',
            'rgba(23, 190, 207, 1)'
        ];

        const traces = [];
        const x = data.epoch;
        let idx = 0;

        for (let parameter_name of Object.keys(data)) {
            if (parameter_name === "epoch") continue;

            const y = data[parameter_name];
            const baseColor = colorway[idx % colorway.length];
            const fadedColor = baseColor.replace(', 1)', ', 0.5)');

            traces.push({
                x: x,
                y: y,
                type: 'scatter',
                mode: 'lines',
                name: parameter_name,
                line: { 
                    width: 2, 
                    color: smooth ? fadedColor : baseColor
                },
                legendgroup: parameter_name,
                showlegend: true
            });

            if (smooth && y.length > 5) {
                const window = Math.max(Math.floor(y.length / 10), 2);
                const smoothedY = [];
                const smoothedX = [];

                for (let i = window - 1; i < y.length; i++) {
                    const slice = y.slice(i - window + 1, i + 1);
                    const avg = slice.reduce((a, b) => a + b, 0) / window;
                    smoothedY.push(avg);
                    smoothedX.push(x[i]);
                }

                traces.push({
                    x: smoothedX,
                    y: smoothedY,
                    type: 'scatter',
                    mode: 'lines',
                    name: `${parameter_name} (trend)`,
                    line: { 
                        width: 2, 
                        color: baseColor 
                    },
                    legendgroup: parameter_name,
                    showlegend: true
                });
            }

            idx++;
        }

        return traces;
    },

    loss_landscape_2d: (data) => {
        const traces = [];

        traces.push({
            x: data.x,
            y: data.y,
            z: data.loss,
            type: 'contour',
            showscale: false,
            contours: {
                coloring: 'none', 
                showlabels: true,
                labelfont: { color: getCSSVar("--text-color") }
            },
            line: {
                color: { color: getCSSVar("--text-color") },
                width: 1
            },
            name: 'Loss Landscape'
        })

        for (let parameter_name of Object.keys(data)) {
            if (["loss", "x", "y"].includes(parameter_name)) {
                continue;
            }

            traces.push({
                x: data[parameter_name].map(t => t[0]),
                y: data[parameter_name].map(t => t[1]),
                mode: 'lines',
                type: 'scatter',
                name: parameter_name
            })

            traces.push({
                x: [data[parameter_name].map(t => t[0])[data[parameter_name].length - 1]],
                y: [data[parameter_name].map(t => t[1])[data[parameter_name].length - 1]],
                mode: 'markers',
                name: `${parameter_name} (final)`,
                marker: {
                    size: 12,
                }
            })
        }

        return traces;
    }
};

const layoutBuilders = {
    base: (title) => ({
        title: { text: title || "Plot"},
        autosize: true,
        paper_bgcolor: getCSSVar("--background-color"),
        plot_bgcolor:  getCSSVar("--background-color"),
        font: { color: getCSSVar("--text-color") },
        xaxis: { gridcolor: getCSSVar("--text-color"), autorange: true },
        yaxis: { gridcolor: getCSSVar("--text-color"), autorange: true },
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

    loss_landscape_2d: (title, data) => {
        const layout = layoutBuilders.base(title || "Loss Landscape")

        return layout;
    },

}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.plotly-container[data-plotly]').forEach(el => {
        const data   = JSON.parse(decodeURIComponent(el.dataset.plotly));
        const title  = el.dataset.title;
        const type   = el.dataset.type;
        const smooth = JSON.parse(decodeURIComponent(el.dataset.smooth));

        const traceBuilder = traceBuilders[type];
        const layoutBuilder = type in layoutBuilders ? layoutBuilders[type] : layoutBuilders.base;
        if (!traceBuilder) {
            console.warn(`[plots.js] Unknown plot type: "${type}"`);
            return;
        }

        Plotly.newPlot(el.id, traceBuilder(data, smooth), layoutBuilder(title), { responsive: true });
    });
});