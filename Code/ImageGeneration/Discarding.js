var figureWidth = 400;  // Height of the entire figure including labels and title.
var figureHeight = 400;  // Width of the entire figure including labels and title.
var figureMargin = {top: 10, right: 10, bottom: 10, left: 10};  // Margin around each individual figure.
var svgWidth = (figureWidth + figureMargin.left + figureMargin.right) * 2;  // Width of the SVG element needed to hold both figures and their padding.
var svgHeight = figureHeight + figureMargin.top + figureMargin.bottom;  // Height of the SVG element needed to hold both figures and their padding.
var svgMargin = {top: 10, right: 10, bottom: 10, left: 10};  // The margin around the set of figures.

// Create the SVG element.
var svg = d3.select("body")
    .append("svg")
        .attr("width", svgWidth + svgMargin.left + svgMargin.right)
        .attr("height", svgHeight + svgMargin.top + svgMargin.bottom)
    .append("g")
        .attr("transform", "translate(" + svgMargin.left + ", " + svgMargin.top + ")");

// Read the data in.
d3.text(dataset, function(text)
    {
        var data = [];
        var rows = text.split("\n");  // Rows of the file.
        var header = rows[0].split("\t");  // The file header.
        var numberClasses = (header.length - 4) / 2;  // Two columns per class, and four unrelated to class.
        for (i = 1; i < rows.length - 1; i++)
        {
            var chunks = rows[i].split("\t");
            var datum = {};
            datum["PatientID"] = parseInt(chunks[0]);
            datum["Class"] = chunks[1];
            for (j = 4; j < 4 + numberClasses; j++)
            {
                var classLabel = header[j].match(/_.*/)[0].substring(1);
                datum[classLabel] = parseFloat(chunks[j]);
            }
            data.push(datum);
        }
        console.log(datum);
    }
);