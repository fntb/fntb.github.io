const builders = {

    scatter: (data) => [{
        x: data.x,
        y: data.y,
        type: 'scatter',
        mode: 'markers',
        opacity: 0.5,
        marker: { color: 'white', size: 4, line: { color: 'rgb(0,0,0)', width: 1 } }
    }],

    // { epochs: [], train: [], val: [] | None }
    learning_curves: (data) => {
        const traces = [{
            x: data.epochs,
            y: data.train,
            type: 'scatter',
            mode: 'lines',
            name: 'Train loss',
            line: { width: 2 }
        }];
        if (data.val) traces.push({
            x: data.epochs,
            y: data.val,
            type: 'scatter',
            mode: 'lines',
            name: 'Val loss',
            line: { width: 2, dash: 'dash' }
        });
        return traces;
    },

    // { epochs: [], grad_norm: [] }
    grad_norms: (data) => [{
        x: data.epochs,
        y: data.grad_norm,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Grad norm',
        line: { width: 2 }
    }],

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

const baseLayout = (title) => ({
    title,
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: { color: '#ccc' },
    xaxis: { gridcolor: '#444' },
    yaxis: { gridcolor: '#444' },
});

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.plotly-container[data-plotly]').forEach(el => {
        const data   = JSON.parse(decodeURIComponent(el.dataset.plotly));
        const title  = el.dataset.title;
        const type   = el.dataset.type ?? 'scatter';

        const builder = builders[type];
        if (!builder) {
            console.warn(`[plots.js] Unknown plot type: "${type}"`);
            return;
        }

        const traces = builder(data);
        Plotly.newPlot(el.id, traces, baseLayout(title), { responsive: true });
    });
});