/**
 * Content Viewer Component
 * Displays the selected document or code content
 */

import { Check, Code, Copy, FileText, Layers } from "lucide-react";
import { Button } from "../../../ui/primitives";
import type { InspectorSelectedItem } from "../../types";

interface ContentViewerProps {
  selectedItem: InspectorSelectedItem | null;
  onCopy: (text: string, id: string) => void;
  copiedId: string | null;
}

export const ContentViewer: React.FC<ContentViewerProps> = ({ selectedItem, onCopy, copiedId }) => {
  if (!selectedItem) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <Layers className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p className="text-sm">Select an item to view</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Content Header - Fixed with proper overflow handling */}
      <div className="p-4 border-b border-white/10 flex items-center gap-3 flex-shrink-0">
        {/* Icon and Metadata - Allow to grow and shrink with min-w-0 for proper truncation */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          {/* Icon - Fixed size, no shrink */}
          <div className="flex-shrink-0">
            {selectedItem.type === "document" ? (
              <FileText className="w-5 h-5 text-cyan-400" />
            ) : (
              <Code className="w-5 h-5 text-green-400" />
            )}
          </div>

          {/* Metadata Content - Can shrink with proper overflow */}
          <div className="min-w-0 flex-1">
            {selectedItem.type === "document" ? (
              <>
                <h4 className="text-sm font-medium text-white/90 truncate">
                  {selectedItem.metadata && "title" in selectedItem.metadata
                    ? selectedItem.metadata.title || "Document"
                    : "Document"}
                </h4>
                {selectedItem.metadata && "section" in selectedItem.metadata && selectedItem.metadata.section && (
                  <p className="text-xs text-gray-500 truncate">{selectedItem.metadata.section}</p>
                )}
              </>
            ) : (
              <>
                <div className="flex items-center gap-2 min-w-0">
                  <span className="px-2 py-0.5 bg-green-500/10 text-green-400 text-xs font-mono rounded flex-shrink-0">
                    {selectedItem.type === "code" && selectedItem.metadata && "language" in selectedItem.metadata
                      ? selectedItem.metadata.language || "unknown"
                      : "unknown"}
                  </span>
                  {selectedItem.type === "code" &&
                    selectedItem.metadata &&
                    "file_path" in selectedItem.metadata &&
                    selectedItem.metadata.file_path && (
                      <span className="text-xs text-gray-500 font-mono truncate min-w-0">
                        {selectedItem.metadata.file_path}
                      </span>
                    )}
                </div>
                {selectedItem.type === "code" &&
                  selectedItem.metadata &&
                  "summary" in selectedItem.metadata &&
                  selectedItem.metadata.summary && (
                    <p className="text-xs text-gray-400 mt-1 line-clamp-2">{selectedItem.metadata.summary}</p>
                  )}
              </>
            )}
          </div>
        </div>

        {/* Copy Button - Never shrinks, always visible */}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => onCopy(selectedItem.content, selectedItem.id)}
          className="text-gray-400 hover:text-white flex-shrink-0"
        >
          {copiedId === selectedItem.id ? (
            <>
              <Check className="w-4 h-4 text-green-400 mr-1.5" />
              <span className="text-xs">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-4 h-4 mr-1.5" />
              <span className="text-xs">Copy</span>
            </>
          )}
        </Button>
      </div>

      {/* Content Body */}
      <div className="flex-1 overflow-y-auto min-h-0 p-6 scrollbar-thin">
        {selectedItem.type === "document" ? (
          <div className="prose prose-invert max-w-none">
            <pre className="whitespace-pre-wrap text-sm text-gray-300 font-sans leading-relaxed">
              {selectedItem.content || "No content available"}
            </pre>
          </div>
        ) : (
          <div className="relative">
            <pre className="bg-black/30 border border-white/10 rounded-lg p-4 overflow-x-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
              <code className="text-sm text-gray-300 font-mono">
                {selectedItem.content || "// No code content available"}
              </code>
            </pre>
          </div>
        )}
      </div>

      {/* Content Footer - Show metadata */}
      <div className="border-t border-white/10 flex-shrink-0">
        <div className="px-4 py-3 flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-4">
            {selectedItem.metadata?.relevance_score != null && (
              <span>
                Relevance:{" "}
                <span className="text-cyan-400">{(selectedItem.metadata.relevance_score * 100).toFixed(0)}%</span>
              </span>
            )}
            {selectedItem.type === "document" && "url" in selectedItem.metadata && selectedItem.metadata.url && (
              <a
                href={selectedItem.metadata.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan-400 hover:text-cyan-300 transition-colors underline"
              >
                View Source
              </a>
            )}
          </div>
          <span className="text-gray-600">{selectedItem.type === "document" ? "Document Chunk" : "Code Example"}</span>
        </div>
      </div>
    </div>
  );
};
